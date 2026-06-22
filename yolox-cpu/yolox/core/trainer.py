#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize,
    get_num_devices
)


class Trainer:
    def __init__(self, exp, args):
        # ==== 基础初始化 ====
        self.exp = exp
        self.args = args

        # ==== 设备检测 ====
        num_gpu = get_num_devices() if args.devices is None else args.devices
        self.device = "cpu" if num_gpu == 0 or not torch.cuda.is_available() else f"cuda:{get_local_rank()}"

        # ==== 分布式训练 ====
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.is_distributed = self.world_size > 1

        # ==== 训练参数 ====
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16 if self.device != "cpu" else False
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_training) if self.device != "cpu" else None
        self.use_model_ema = exp.ema
        self.data_type = torch.float16 if args.fp16 and self.device != "cpu" else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # ==== 日志系统 ====
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

        # 添加调试信息
        logger.info(f'Training on {self.device.upper()}, num_gpu={num_gpu}')
        if self.device == "cpu":
            logger.warning("Using CPU for training. This will be much slower than GPU training.")
        else:
            logger.info(f"Using GPU {self.device} for training.")

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            self.after_train()

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # 模型初始化
        model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)

        # 优化器
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # 恢复训练
        model = self.resume_train(model)

        # 数据加载
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )

        # 数据预取
        if self.device != "cpu":
            logger.info("init prefetcher, this might take one minute or less...")
            self.prefetcher = DataPrefetcher(self.train_loader)
        else:
            self.loader = iter(self.train_loader)

        self.max_iter = len(self.train_loader)

        # 学习率调度
        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )

        # 分布式训练
        if self.is_distributed and self.device != "cpu":
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        # EMA模型
        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        # 评估器
        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )

        # TensorBoard
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        # 数据加载
        if self.device == "cpu":
            inps, targets, _, _ = next(self.loader)
            inps = inps.to(self.device)
            targets = targets.to(self.device)
        else:
            inps, targets = self.prefetcher.next()
            inps = inps.to(self.data_type)
            targets = targets.to(self.data_type)

        # 数据预处理
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        # 前向传播
        if self.device != "cpu" and self.amp_training:
            with torch.amp.autocast('cuda', enabled=True):
                outputs = self.model(inps, targets)
        else:
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        # 反向传播
        self.optimizer.zero_grad()
        if self.device != "cpu" and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # EMA更新
        if self.use_model_ema:
            self.ema_model.update(self.model)

        # 学习率更新
        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # 记录指标
        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_epoch(self):
        logger.info(f"---> start train epoch {self.epoch + 1}")
        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("---> No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("---> Add additional L1 loss now!")
            if is_parallel(self.model):
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        if (self.iter + 1) % self.exp.print_interval == 0:
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()])

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()])

            # 添加设备信息到日志
            device_info = f"device: {self.device}"
            logger.info(
                "{}, {}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    device_info,
                    gpu_mem_usage() if self.device != "cpu" else 0,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        if self.device != "cpu" and (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.start_epoch = ckpt.get("start_epoch", 0)
            logger.info(f"loaded checkpoint '{ckpt_file}' (epoch {self.start_epoch})")
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt = torch.load(self.args.ckpt, map_location=self.device)
                model = load_ckpt(model, ckpt["model"] if "model" in ckpt else ckpt)
            self.start_epoch = 0
        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        # 确保模型在正确的设备上
        if self.device == "cpu":
            evalmodel = evalmodel.cpu()

        ap50_95, ap50, summary = self.exp.eval(
            evalmodel, self.evaluator, self.is_distributed
        )
        self.model.train()
        if self.rank == 0:
            self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
            self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", ap50_95 > self.best_ap)
        self.best_ap = max(self.best_ap, ap50_95)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )
