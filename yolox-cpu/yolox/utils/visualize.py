#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. and All rights reserved.
#用于在图片上画出检测框、中文标签、故障等级 （画出检测框 + 显示中文标签 + 按故障等级（高 / 中 / 低）显示不同颜色 + 显示置信度。）
#显示中文
# 显示故障等级（高 / 中 / 低）
# 用红 / 橙 / 绿区分严重程度
# 解决 OpenCV 不支持中文乱码问题 把OpenCV图片转为PIL图片（PIL支持中文显示）
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["vis", "vis_chinese"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    """
    原始可视化函数（英文标签）
    """
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def vis_chinese(img, boxes, scores, cls_ids, fault_levels, conf=0.5, class_names=None, fault_location_names=None):
    """
    中文可视化函数，支持故障等级显示
    使用PIL库绘制中文，避免OpenCV不支持中文的问题
    
    参数:
        img: 图像数组 (BGR格式)
        boxes: 边界框列表 [N, 4]
        scores: 置信度列表 [N]
        cls_ids: 类别ID列表 [N]
        fault_levels: 故障等级列表 [N] ("high", "middle", "low")
        conf: 置信度阈值
        class_names: 英文类别名称列表
        fault_location_names: 故障位置中文名称映射字典
    """
    # 将OpenCV图像(BGR)转换为PIL图像(RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体，如果失败则使用默认字体
    try:
        # Windows系统字体路径
        font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
        font = ImageFont.truetype(font_path, 20)
        font_small = ImageFont.truetype(font_path, 14)
    except:
        try:
            font_path = "C:/Windows/Fonts/simsun.ttc"  # 宋体
            font = ImageFont.truetype(font_path, 20)
            font_small = ImageFont.truetype(font_path, 14)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        fault_level = fault_levels[i] if i < len(fault_levels) else "low"
        
        if score < conf:
            continue
            
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        
        # 获取故障位置中文名称
        english_name = class_names[cls_id]
        chinese_name = fault_location_names.get(english_name, english_name)
        
        # 等级颜色映射：红色(高)、橙色(中)、绿色(低)
        # PIL使用RGB格式，需要转换
        if fault_level == "high":
            level_color = (255, 0, 0)      # 红色 - 高等级
            level_text = "高"
        elif fault_level == "middle":
            level_color = (255, 165, 0)     # 橙色 - 中等级
            level_text = "中"
        else:
            level_color = (0, 255, 0)      # 绿色 - 低等级
            level_text = "低"
        
        # 组合显示文本：位置+等级
        display_text = f"{chinese_name}({level_text}) {score*100:.1f}%"
        
        # 绘制边界框（根据等级使用不同颜色）
        draw.rectangle([(x0, y0), (x1, y1)], outline=level_color, width=2)
        
        # 计算文本大小
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 绘制文本背景
        draw.rectangle(
            [(x0, y0 - text_height - 10), (x0 + text_width + 10, y0)],
            fill=level_color
        )
        
        # 绘制文本
        draw.text(
            (x0 + 5, y0 - text_height - 5),
            display_text,
            fill=(255, 255, 255),
            font=font
        )
        
        # 在框下方标注等级
        level_display = f"[等级: {level_text}]"
        level_bbox = draw.textbbox((0, 0), level_display, font=font_small)
        level_width = level_bbox[2] - level_bbox[0]
        level_height = level_bbox[3] - level_bbox[1]
        draw.text(
            (x0 + 5, y1 + 5),
            level_display,
            fill=level_color,
            font=font_small
        )
    
    # 将PIL图像转换回OpenCV格式(BGR)
    result_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return result_img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
