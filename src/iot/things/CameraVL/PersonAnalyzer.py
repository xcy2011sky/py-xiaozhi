import sys
import os
sys.path.append(os.path.abspath("d:/ws/Agent/py-xiaozhi"))
import time
import logging
from typing import BinaryIO
import threading
from enum import Enum
from src.iot.things.CameraVL.FaceAnalyzer import FaceAnalyzer


class FaceObject:
    def __init__(self, face_id: int,entity_id:str, confidence: float, rect: dict):
        self.face_id = face_id
        self.entity_id = entity_id  
        self.confidence = confidence
        self.rect = rect
       


class PersonObject:
    def __init__(self, face_object: FaceObject, hand_gesture: str, age: int, gender: str, emotion: str):    
        self.face_object = face_object 
        self.name = face_object.entity_id if face_object else "Unknown"  # Default name if no face object is provided
        self.hand_gesture = hand_gesture  # Static hand gesture analysis result
        self.age = age  # Age analysis result 
        self.gender = gender    # Gender analysis result
        self.emotion = emotion  # Emotion analysis result
        self.confidence = face_object.confidence if face_object else 0.0  # Confidence score from face analysis



    def __str__(self):
        return (f"PersonObject(name={self.name}, "
                f"face_id={self.face_object.face_id if self.face_object else None}, "
                f"hand_gesture={self.hand_gesture}, "
                f"age={self.age}, " 
                f"gender={self.gender}, "
                f"confidence={self.confidence}, "
                f"emotion={self.emotion})")
    
    

class PersonAnalyzer:

    def __init__(self):
        self.face_analyzer = FaceAnalyzer.get_instance()
        self.logger = logging.getLogger(self.__class__.__name__)

    def analyze(self, image: BinaryIO):

        start_time = time.time()
        persons=[]
        person_faces = []
        # 执行人脸分析、人脸属性、表情和静态手势分析的实际逻辑
        # stage 1 人脸分析： {'Data': {'MatchList': [{'FaceItems': [{'Confidence': 100.0, 'DbName': 'default', 'EntityId': 'yan', 'FaceId': '262990250', 'Score': 1.0}, {'Confidence': 69.59747, 'DbName': 'default', 'EntityId': 'yan2', 'FaceId': '263620824', 'Score': 0.4881063997745514}], 'Location': {'Height': 129, 'Width': 99, 'X': 140, 'Y': 58}, 'QualitieScore': 99.6685}]}, 'RequestId': '5D8E26F5-DC7A-5D29-BE93-E9A39449F746'}
        #logging.info("stage 1: Starting person analysis...")
        self.face_analyzer=FaceAnalyzer.get_instance()
        result = self.face_analyzer.search_face(image)
        face_result = result['result']
        if result['status'] =='success':
            # 遍历多个人脸对象列表
            for match_info in face_result.data.match_list:
                face_objects = []
                face_items = match_info.face_items
                # 遍历每个FaceItem，创建FaceObject实例      
                for face_item in face_items:
                    face_object = FaceObject(
                        face_id=face_item.face_id,
                        entity_id=face_item.entity_id,
                        confidence=face_item.confidence,
                        rect={}
                    )
                    face_objects.append(face_object)
            # 选择置信度最高的人脸对象
                if face_objects:
                    best_face_object = max(face_objects, key=lambda f: f.confidence)
                    # 创建PersonObject实例
                    self.face_object = best_face_object
                    self.face_object.rect = match_info.location
                    person_faces.append(self.face_object)
            # 打印每个人脸对象的信息
            face_search_time = time.time()
            #logging.info(f"stage 1: Detected faces: {len(person_faces)} Face search took: {face_search_time - start_time:.2f}seconds" )
        else:
            self.logger.warning("stage 1:没有检测到人脸信息")
            
                 
        # state2  添加人脸属性分析
        # {'Data': {'AgeList': [26, 20], 'BeautyList': [], 'DenseFeatureLength': 0, 'DenseFeatures': [], 'Expressions': [], 'FaceCount': 2, 'FaceProbabilityList': [0.95, 0.95], 'FaceRectangles': [171, 85, 46, 57, 310, 66, 47, 61], 'GenderList': [1, 0], 'Glasses': [], 'HatList': [], 'LandmarkCount': 0, 'Landmarks': [], 'Masks': [], 'PoseList': [], 'Pupils': [], 'Qualities': {'BlurList': [], 'FnfList': [], 'GlassList': [], 'IlluList': [], 'MaskList': [], 'NoiseList': [], 'PoseList': [], 'ScoreList': []}}, 'RequestId': 'BE15BC11-36D3-5F45-B789-DD2CF9ABFFA2'}
        result = self.face_analyzer.recognize_face(image)
        face_attribers = result['result']
        status = result['status']
        #logging.info("stage 2: Starting face attribute analysis...")
        if status =='success':
            # 遍历人脸属性分析结果
           
            for face in person_faces:
                # 获取对应人脸的属性
                index = person_faces.index(face)
                age = face_attribers.data.age_list[index] if index < len(face_attribers.data.age_list) else None
                age=age if age is not None else 0  # 默认年龄为0
                gender = face_attribers.data.gender_list[index] if index < len(face_attribers.data.gender_list) else None
                if gender == 1:
                    gender = "male"
                elif gender == 0:   
                    gender = "female"
                else:
                    gender = "unknown"
                person=PersonObject(
                    face_object=face,
                    hand_gesture="",  
                    age=age,      
                    gender=gender,                
                    emotion=""  
                )
                persons.append(person)
            face_attribers_time = time.time()
            #logging.info(f"stage 2:Face attribute analysis completed Detected persons: {len(persons)}, Face attribute analysis took: {face_attribers_time - face_search_time:.2f} seconds")
        else:
            self.logger.warning("stage 2:没有检测到人脸属性信息")

        # state3 人脸表情分析
        # {'Data': {'Elements': [{'Expression': 'happiness', 'FaceProbability': 0.822265625, 'FaceRectangle': {'Height': 255, 'Left': 344, 'Top': 122, 'Width': 189}}]}, 'RequestId': '6C0F6604-7996-54C3-9E70-B8157F081A9E'}
        result = self.face_analyzer.recongize_expression(image)
        face_emotions = result['result']
        status = result['status']
        #logging.info("stage 3: Starting face emotion analysis...")
        if status =='success':
            # 遍历人脸表情分析结果
            for person in persons:
                # 获取对应人脸的表情
                index = persons.index(person)
                emotion = face_emotions.data.elements[index].expression if index < len(face_emotions.data.elements) else None
                if emotion is None:
                    emotion = "unknown"
                else:
                    person.emotion = emotion
            face_emotions_time = time.time()
            #logging.info(f"stage 3: Detected emotions: {len(face_emotions.data.elements)}, Face emotion analysis took: {face_emotions_time - face_attribers_time:.2f} seconds")
        else:
            self.logger.warning("stage 3:没有检测到人脸表情信息")

        # state4 静态手势分析,当前只能识别一个人静态手势，效果不太好
        # {'Data': {'Height': 337, 'Score': 0.8125, 'Type': 'good', 'Width': 164, 'X': 107, 'Y': 169}, 'RequestId': '3E0D1427-84A8-5844-8350-C24B4AA12905'}
        result = self.face_analyzer.recognize_hand_gesture(image)
        hand_gesture = result['result']
        status = result['status']
        #logging.info("stage 4: Starting static hand gesture analysis...")
        if status =='success': 
            # 获取静态手势分析结果
            for person in persons:
                # 假设只分析第一个人
                person.hand_gesture = hand_gesture.data.type if hand_gesture.data.type else "unknown"
            hand_gesture_time = time.time()
            #logging.info(f"stage 4:Detected static hand gestures: {len(persons)}, Static hand gesture analysis took: {hand_gesture_time - face_emotions_time:.2f} seconds", )   

        else:
            self.logger.warning("stage 4:没有检测到静态手势信息")
            
        # state5 打印所有人的信息
        #logging.info("stage 5: Finalizing person analysis...")
        for person in persons:
            self.logger.info(str(person))
        # 记录结束时间
        end_time = time.time()
        # 打印分析耗时  
        self.logger.info(f"stage 5:Analysis completed in {end_time - start_time:.2f} seconds") 
        return persons