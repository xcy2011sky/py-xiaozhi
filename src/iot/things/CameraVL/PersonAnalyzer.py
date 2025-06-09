from typing import BinaryIO
import threading
from enum import Enum

from iot.things.CameraVL.FaceAnalyzer import FaceAnalyzer


class FaceObject:
    def __init__(self, face_id: int,entity_id:str, confidence: float, rect: dict):
        self.face_id = face_id
        self.entity_id = entity_id  # Assuming EntityId is the same as face_id
        self.confidence = confidence
        self.rect = rect
       

class PersonObject:
    PersonObjects=[]
    def __init__(self,
                 face_object: FaceObject,#只保存Confidence最高的人脸信息
                 hand_gesture: str,
                 age: int,
                 gender: str,
                 emotion: str):
        self.face_object = face_object
        self.hand_gesture = hand_gesture
        self.age = age
        self.gender = gender
        self.emotion = emotion
        self.face_analyzer = None  # Placeholder for FaceAnalyzer instance
        self.name = face_object.entity_id;  # Placeholder for person's name, if available
    def __str__(self):
        return (f"PersonObject(face_id={self.face_object.face_id}, "
                f"confidence={self.face_object.confidence}, "
                f"rect={self.face_object.rect}, "
                f"hand_gesture={self.hand_gesture}, "
                f"age={self.age}, "   
                f"gender={self.gender}, "
                f"emotion={self.emotion})")    
    def analyze(self, image: BinaryIO):
        analysis_thread = threading.Thread(target=self._perform_analysis, args=(image,))
        analysis_thread.start()
        analysis_thread.join()  # 等待线程完成
        # 返回自身作为占位符，实际应根据分析结果创建并返回一个PersonObject实例

    def _perform_analysis(self, image: BinaryIO):
        # 执行人脸分析、人脸属性、表情和静态手势分析的实际逻辑
        # stage 1 人脸分析： {'Data': {'MatchList': [{'FaceItems': [{'Confidence': 100.0, 'DbName': 'default', 'EntityId': 'yan', 'FaceId': '262990250', 'Score': 1.0}, {'Confidence': 69.59747, 'DbName': 'default', 'EntityId': 'yan2', 'FaceId': '263620824', 'Score': 0.4881063997745514}], 'Location': {'Height': 129, 'Width': 99, 'X': 140, 'Y': 58}, 'QualitieScore': 99.6685}]}, 'RequestId': '5D8E26F5-DC7A-5D29-BE93-E9A39449F746'}
        self.face_analyzer=FaceAnalyzer.get_instance()
        status,face_result = self.face_analyzer.search_face(image)
        if status =="success":
            # 遍历多个人脸对象列表
            person_faces = []
            for match_info in face_result.data.MatchList:
                face_objects = []
                face_items = match_info.FaceItems
                # 遍历每个FaceItem，创建FaceObject实例      
                for face_item in face_items:
                    face_object = FaceObject(
                        face_id=face_item.FaceId,
                        entity_id=face_item.EntityId,
                        confidence=face_item.confidence,
                        rect={}
                    )
                    face_objects.append(face_object)
            # 选择置信度最高的人脸对象
                if face_objects:
                    best_face_object = max(face_objects, key=lambda f: f.confidence)
                    # 创建PersonObject实例
                    self.face_object = best_face_object
                    self.face_object.rect = match_info.Location
                    person_faces.append(self.face_object)
            # 打印每个人脸对象的信息
            for person_face in person_faces:
                print(f"Detected Face: {person_face.face_id}, "
                      f"Confidence: {person_face.confidence}, "
                      f"Rect: {person_face.rect}")          
            
                 
        # state2  添加人脸属性分析
        # {'Data': {'AgeList': [26, 20], 'BeautyList': [], 'DenseFeatureLength': 0, 'DenseFeatures': [], 'Expressions': [], 'FaceCount': 2, 'FaceProbabilityList': [0.95, 0.95], 'FaceRectangles': [171, 85, 46, 57, 310, 66, 47, 61], 'GenderList': [1, 0], 'Glasses': [], 'HatList': [], 'LandmarkCount': 0, 'Landmarks': [], 'Masks': [], 'PoseList': [], 'Pupils': [], 'Qualities': {'BlurList': [], 'FnfList': [], 'GlassList': [], 'IlluList': [], 'MaskList': [], 'NoiseList': [], 'PoseList': [], 'ScoreList': []}}, 'RequestId': 'BE15BC11-36D3-5F45-B789-DD2CF9ABFFA2'}
        status,face_attribers = self.face_analyzer.recognize_face(image)
        if status == "success":
            # 遍历人脸属性分析结果
            persons=[]
            for face in person_faces:
                # 获取对应人脸的属性
                index = person_faces.index(face)
                age = face_attribers.data.AgeList[index] if index < len(face_attribers.data.AgeList) else None
                age=age if age is not None else 0  # 默认年龄为0
                gender = face_attribers.data.GenderList[index] if index < len(face_attribers.data.GenderList) else None
                if gender == 1:
                    gender = "male"
                elif gender == 0:   
                    gender = "female"
                else:
                    gender = "unknown"
                person=PersonObject(
                    face_object=face,
                    hand_gesture="",  # 静态手势分析结果待添加
                    age=age,    # 年龄分析结果待添加        
                    gender=gender,  # 性别分析结果待添加                
                    emotion=""  # 表情分析结果待添加    
                )
                persons.append(person)
            # 打印每个人的属性信息  
            for person in persons:
                print(person)
        else:
            print("没有检测到人脸属性信息")

        # state3 人脸表情分析
        # {'Data': {'Elements': [{'Expression': 'happiness', 'FaceProbability': 0.822265625, 'FaceRectangle': {'Height': 255, 'Left': 344, 'Top': 122, 'Width': 189}}]}, 'RequestId': '6C0F6604-7996-54C3-9E70-B8157F081A9E'}
        status,face_emotions = self.face_analyzer.recongize_expression(image)
        if status == "success":
            # 遍历人脸表情分析结果
            for face in person_faces:
                # 获取对应人脸的表情
                index = person_faces.index(face)
                emotion = face_emotions.data.Elements[index].Expression if index < len(face_emotions.data.Elements) else None
                if emotion is None:
                    emotion = "unknown"
                else:
                    persons[index].emotion = emotion
            # 打印每个人的表情信息
            for person in persons:  
                print(f"{person.name} 的表情是 {person.emotion}")  
        else:
            print("没有检测到人脸表情信息")

        # state4 静态手势分析,当前只能识别一个人静态手势，效果不太好
        # {'Data': {'Height': 337, 'Score': 0.8125, 'Type': 'good', 'Width': 164, 'X': 107, 'Y': 169}, 'RequestId': '3E0D1427-84A8-5844-8350-C24B4AA12905'}
        status,hand_gesture = self.face_analyzer.recognize_hand_gesture(image)
        if status == "success": 
            # 获取静态手势分析结果
            for person in persons:
                # 假设只分析第一个人
                person.hand_gesture = hand_gesture.data.Type if hand_gesture.data.Type else "unknown"
        else:
            print("没有检测到静态手势信息")
            
        # state5 打印所有人的信息
        for person in persons:
            print(person)
        PersonObject.PersonObjects.clear()
        PersonObject.PersonObjects=persons

    
