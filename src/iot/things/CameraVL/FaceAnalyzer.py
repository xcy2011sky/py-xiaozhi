import json
import os
import sys
import cv2
import asyncio
import numpy as np
from typing import List, BinaryIO
import logging  
import io  
from enum import Enum


# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

from alibabacloud_facebody20191230.client import Client as facebody20191230Client
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_facebody20191230 import models as facebody_20191230_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient
import threading

class HandGesture(Enum):
    """静态手势识别结果的枚举类型"""
    BG = 'bg'     # 无法识别的手势
    OK = 'ok'     # 确认
    PALM = 'palm' # 手掌
    LEFT = 'left' # 握拳且大拇指向左
    RIGHT = 'right' # 握拳且大拇指向右
    GOOD = 'good'   # 点赞（即握拳且大拇指向上）
    MUTE = 'mute'   # 噤声（将食指放在嘴上即被识别为噤声）
    DOWN = 'down'   # 握拳且大拇指向下

    @classmethod
    def from_string(cls, value: str) -> 'HandGesture':
        """从字符串转换为枚举值"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"无效的手势类型: {value}. 可选值: {[e.value for e in cls]}")

class FaceExpression(Enum):
    """人脸表情的枚举类型"""
    NEUTRAL = 'neutral'     # 中性
    HAPPINESS = 'happiness' # 高兴
    SURPRISE = 'surprise'   # 惊讶
    SADNESS = 'sadness'     # 伤心
    ANGER = 'anger'         # 生气
    DISGUST = 'disgust'     # 厌恶
    FEAR = 'fear'           # 害怕
    POUTY = 'pouty'         # 嘟嘴
    GRIMACE = 'grimace'     # 鬼脸

    @classmethod
    def from_string(cls, value: str) -> 'FaceExpression':
        """从字符串转换为枚举值"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"无效的表情类型: {value}. 可选值: {[e.value for e in cls]}")


class FaceAnalyzer:
    _instance = None
    _lock = threading.Lock()
    client = None

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.client:
            self.init_client()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def create_client(self) -> facebody20191230Client:
        try:
            credential = CredentialClient()
            config = open_api_models.Config(
                credential=credential
            )
            # Endpoint 请参考 https://api.aliyun.com/product/facebody
            config.endpoint = f'facebody.cn-shanghai.aliyuncs.com'
            client = facebody20191230Client(config)
            self.logger.info("client创建成功...")
            return client
        except Exception as error:
            self.logger.error(f'创建client失败：{str(error)}', exc_info=True)
            raise

    def init_client(self):
        try:
            self.client = self.create_client()
        except Exception as error:
            self.logger.error("初始化客户端失败: %s", error, exc_info=True)
            raise

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def recongize_expression(self, image:BinaryIO) -> dict:
        try:
            recognize_expression_request = facebody_20191230_models.RecognizeExpressionAdvanceRequest(
                image_urlobject=image,  
            )
            runtime = util_models.RuntimeOptions()
            response = self.client.recognize_expression_advance(recognize_expression_request, runtime)
            result = {
                'status': 'success',
                'result': response.body
            }
            #self.logger.info("表情分析完成，结果: %s", result['result'])
            return result
        except Exception as error:
            #self.logger.error(f"分析过程中发生错误: {error}", exc_info=True)
            return {
                'status': 'error',
                'result': str(error)
            }

    def search_face(self, image:BinaryIO) -> dict:
     
        try:

            search_face_request = facebody_20191230_models.SearchFaceAdvanceRequest(
                image_url_object=image,  
                limit=2,
                db_name="default"  
            )
            runtime = util_models.RuntimeOptions()
            response = self.client.search_face_advance(search_face_request, runtime)
            #self.logger.info("response: %s", response.body)

            result = {
                'status': 'success',
                'result': response.body
            }
            return result
        except Exception as error:
            #self.logger.error(f"搜索人脸时发生错误: {error}", exc_info=True)
            result = {
                'status': 'error',
                'result': str(error)
            }
            return result

    def recognize_face(self, image:BinaryIO) -> dict:
        try:
            recognize_face_request = facebody_20191230_models.RecognizeFaceAdvanceRequest(
                 age=True,
                 gender=True,
                 max_face_number=5,
                 quality=False,
                 image_urlobject=image,
            )
            runtime = util_models.RuntimeOptions()
            response = self.client.recognize_face_advance(recognize_face_request, runtime)
            result = {
                'status': 'success',
                'result': response.body
            }
            #self.logger.info("人脸属性分析完成，结果: %s", result['result'])
            return result
        except Exception as error:
            #self.logger.error("分析人脸属性时发生错误: %s", error, exc_info=True)
            result = {
                'status': 'error',
                'result': str(error)
            }
            return result
           

    def recognize_hand_gesture(self, image:BinaryIO) -> dict:
        try:

            recognize_hand_gesture_request = facebody_20191230_models.RecognizeHandGestureAdvanceRequest(
                app_id="gesture_app",
                gesture_type="gesture_recognition",
                image_urlobject=image,  
            )
            runtime = util_models.RuntimeOptions()

            response = self.client.recognize_hand_gesture_advance(recognize_hand_gesture_request, runtime)

            result = {
                'status': 'success',
                'result': response.body 
            }
            #self.logger.info("手势识别完成，结果: %s", result['result'])    
            return result
        except Exception as error:
            #self.logger.error(f"手势识别时发生错误: {error}", exc_info=True)
            result = {
                'status': 'error',
                'result': str(error)
            }
            return result
           

  
        

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath("d:/ws/Agent/py-xiaozhi"))
    import asyncio
    import cv2
    import base64
    #from src.iot.things.CameraVL.Camera import Camera
    
    # 创建人脸分析器实例

    
    face_analyzer = FaceAnalyzer.get_instance()
    # img = open(r'assets/data/hand_only.jpg', 'rb')
    # result=face_analyzer.recognize_hand_gesture(img)
   
    
    
    # 创建摄像头实例
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("无法打开摄像头")
        exit(1)

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    
    # 创建显示窗口
    cv2.namedWindow("Camera Preview", cv2.WINDOW_AUTOSIZE)
    
    # 用于控制分析频率（1FPS）
    frame_count = 0
    
    try:
        while True:
            # 读取一帧画面

            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Camera Preview", frame)
            
            if frame_count % 60 == 0:
                # 将帧转换为 JPEG 格式并编码为base64
                _, buffer = cv2.imencode(".jpg", frame)
                
                # 确保buffer是numpy数组
                buffer_array = io.BytesIO(buffer.tobytes())
                
                # 并行执行三种分析
                loop = asyncio.get_event_loop()
                
                # 使用run_in_executor来并行执行阻塞操作
                face_detect_task = loop.run_in_executor(None, face_analyzer.recognize_hand_gesture, buffer_array)

                
                # 等待所有分析完成
                try:
                    # 直接获取结果
                    #face_detect_result = loop.run_until_complete(face_detect_task)
                    attribute_result = loop.run_until_complete(face_detect_task)

                except Exception as e:
                    print(f"分析过程中发生错误: {e}")
    
            
            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 增加帧计数器
            frame_count += 1
            #print(f"当前帧数: {frame_count}")
            
    except KeyboardInterrupt:
        print("程序被用户中断")
    
    finally:
        # 释放资源
        cv2.destroyAllWindows()