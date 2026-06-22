#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# 故障位置类别定义（模型只检测位置，等级在推理时判断）
VOC_CLASSES = (
    "sensor_damage",    # 传感器损坏
    "red_warning",       # 红灯预警
)

# 故障等级阈值配置（可根据实际情况调整）
# 根据检测框面积大小判断等级
FAULT_LEVEL_CONFIG = {
    "sensor_damage": {
        "high_area_threshold": 15000,    # 高等级面积阈值（降低）
        "middle_area_threshold": 8000,   # 中等级面积阈值（降低）
    },
    "red_warning": {
        "high_area_threshold": 6000,     # 红灯通常较小，阈值更低
        "middle_area_threshold": 3000,
    }
}

# 故障位置中文名称映射
FAULT_LOCATION_NAMES = {
    "sensor_damage": "传感器损坏",
    "red_warning": "红灯预警",
}

# 故障等级中文名称
FAULT_LEVEL_NAMES = {
    "high": "高",
    "middle": "中",
    "low": "低",
}

def calculate_fault_level(fault_type, bbox_area, img_area):
    """
    根据检测框面积和图像面积比例判断故障等级
    
    参数:
        fault_type: 故障类型 (sensor_damage/red_warning)
        bbox_area: 检测框面积 (像素)
        img_area: 图像总面积 (像素)
    
    返回:
        fault_level: 故障等级 ("high", "middle", "low")
    """
    # 计算面积占比
    area_ratio = bbox_area / img_area if img_area > 0 else 0
    
    # 获取配置阈值
    config = FAULT_LEVEL_CONFIG.get(fault_type, {})
    high_threshold = config.get("high_area_threshold", 15000)
    middle_threshold = config.get("middle_area_threshold", 8000)
    
    # 根据面积判断等级
    if bbox_area >= high_threshold:
        return "high"
    elif bbox_area >= middle_threshold:
        return "middle"
    else:
        return "low"

def get_fault_description(fault_type, fault_level):
    """
    获取故障描述（中文）
    
    参数:
        fault_type: 故障类型
        fault_level: 故障等级
    
    返回:
        description: 故障描述字符串
    """
    location_name = FAULT_LOCATION_NAMES.get(fault_type, fault_type)
    level_name = FAULT_LEVEL_NAMES.get(fault_level, fault_level)
    return f"{location_name}({level等级})"

def parse_bbox_area(bbox):
    """
    计算检测框面积
    
    参数:
        bbox: [x1, y1, x2, y2] 格式的边界框
    
    返回:
        area: 面积 (像素)
    """
    if len(bbox) >= 4:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return max(0, width * height)
    return 0
