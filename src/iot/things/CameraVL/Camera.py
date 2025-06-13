import asyncio
import base64
import io
import logging
import threading

import cv2


from src.application import Application
from src.constants.constants import DeviceState
from src.iot.thing import Thing
from src.iot.things.CameraVL import VL
from src.iot.things.CameraVL.PersonAnalyzer import PersonAnalyzer, PersonObject

logger = logging.getLogger("Camera")


class Camera(Thing):
    def __init__(self):
        super().__init__("Camera", "摄像头管理")
        self.app = None
        """初始化摄像头管理器"""
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        # 加载配置
        self.cap = None
        self.is_running = False
        self.camera_thread = None
        self.person_analyzer_thread = None
        self.result = ""
        from src.utils.config_manager import ConfigManager

        self.config = ConfigManager.get_instance()
        # 摄像头控制器
        VL.ImageAnalyzer.get_instance().init(
            self.config.get_config("CAMERA.VLapi_key"),
            self.config.get_config("CAMERA.Loacl_VL_url"),
            self.config.get_config("CAMERA.models"),
        )
        self.VL = VL.ImageAnalyzer.get_instance()
        logger.info(f"[虚拟设备] 摄像头设备初始化完成")

        self.add_property_and_method()  # 定义设备方法与状态属性

        # 启动实时人物分析
        self.person_analyzer_thread = None
        self.enable_person_analyzer = True
        self.is_person_analyzer_running = False
        self.person_analyzer = PersonAnalyzer()
        self.person_list_holder = []  # 用于存储分析结果
        logger.info(f"[虚拟设备] 视觉实时人物分析初始化完成")
 
      

    def add_property_and_method(self):
        # 定义属性
        self.add_property("power", "摄像头是否打开", lambda: self.is_running)
        self.add_property("result", "识别画面的内容", lambda: self.result)
        # 定义方法
        self.add_method(
            "start_camera", "打开摄像头", [], lambda params: self.start_camera()
        )

        self.add_method(
            "stop_camera", "关闭摄像头", [], lambda params: self.stop_camera()
        )

        self.add_method(
            "capture_frame_to_base64",
            "识别画面",
            [],
            lambda params: self.capture_frame_to_base64(),
        )

    def _camera_loop(self):
        """摄像头线程的主循环"""
        camera_index = self.config.get_config("CAMERA.camera_index")
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            logger.error("无法打开摄像头")
            return

        # 设置摄像头参数
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH, self.config.get_config("CAMERA.frame_width")
        )
        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT, self.config.get_config("CAMERA.frame_height")
        )
        self.cap.set(cv2.CAP_PROP_FPS, self.config.get_config("CAMERA.fps"))

        self.is_running = True
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("无法读取画面")
                break
            if not self.is_person_analyzer_running and self.enable_person_analyzer:
                logger.info("启动实时人物分析线程")
                self.start_person_analyze()
                self.is_person_analyzer_running = True
                

            # 显示画面
            cv2.imshow("Camera", frame)
            # 按下 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.is_running = False

        # 释放摄像头并关闭窗口
        self.cap.release()
        cv2.destroyAllWindows()

    def start_camera(self):
        """启动摄像头线程"""
        if self.camera_thread is not None and self.camera_thread.is_alive():
            logger.warning("摄像头线程已在运行")
            return

        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        logger.info("摄像头线程已启动")
        return {"status": "success", "message": "摄像头线程已打开"}

    def capture_frame_to_base64(self):
        """截取当前画面并转换为 Base64 编码"""
        if not self.cap or not self.cap.isOpened():
            logger.error("摄像头未打开")
            return None

        ret, frame = self.cap.read()
        if not ret:
            logger.error("无法读取画面")
            return None

        # 将帧转换为 JPEG 格式
        _, buffer = cv2.imencode(".jpg", frame)

        # 将 JPEG 图像转换为 Base64 编码
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        self.result = str(self.VL.analyze_image(frame_base64))
        # 获取应用程序实例
        self.app = Application.get_instance()
        logger.info("画面已经识别到啦")
        self.app.set_device_state(DeviceState.LISTENING)
        asyncio.create_task(self.app.protocol.send_wake_word_detected("播报识别结果"))
        return {"status": "success", "message": "识别成功", "result": self.result}
    
    def _person_analyzer_loop(self):
        
        """实时人物分析线程的主循环"""
        if not self.cap or not self.cap.isOpened(): 
            logger.error("摄像头未打开")
            return
        frame_count = 0
        old_persons=[]

        while self.is_running:
            if frame_count % 60 == 0:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("无法读取画面")
                    break

                # 将帧转换为 JPEG 格式
                _, buffer = cv2.imencode(".jpg", frame)
                buffer_array = io.BytesIO(buffer.tobytes())
                if self.person_analyzer is None:
                    raise ValueError("PersonAnalyzer未初始化")
                
                persons=self.person_analyzer.analyze(buffer_array)
                # for person in persons:
                #     logger.info(f"识别到人: {person}")
                if self.compare_person_list(persons,old_persons)==False:  
                    self.person_chat_message(persons)
                    old_persons=persons
                else:
                    logger.info("人物分析结果无变化")

            else:
                result = None
            frame_count += 1
     
        logger.warning(f" 实时人物分析线程已停止")
    def compare_person(self,person1:PersonObject,person2:PersonObject)->bool:
        return person1.name==person2.name and person1.gender==person2.gender and person1.emotion==person2.emotion and person1.hand_gesture==person2.hand_gesture
    
    def compare_person_list(self,person_list1:list[PersonObject],person_list2:list[PersonObject])->bool:
        if(len(person_list1)!=len(person_list2)):
            return False
        else:
            for person1 in person_list1:
                for person2 in person_list2:
                    if(self.compare_person(person1,person2)):
                        return True
                    else: 
                        return False
            return False
    
    
    def person_chat_message(self,person_list:list[PersonObject]):
        """
        处理人物分析结果，生成聊天消息
        :param person_list: 人物对象列表
        """
        chat_message= ""
        self.app = Application.get_instance()
        if person_list.__len__() ==0  and   self.person_list_holder.__len__() == 0:
            logger.info("初次启动没有识别到人物")
            chat_message = "你好，当前场景无人"
            self.app._on_incoming_vision(chat_message)
    
            return
        if  person_list.__len__() > 0  and self.person_list_holder.__len__() == 0:
            self.person_list_holder = person_list
            logger.info(f"新识别到人物: {self.person_list_holder}")
            is_yan = False
            is_other = False
            for person in person_list:
                if person.name =="yan2":
                    is_yan = True   
                else:
                    is_other = True
            if is_yan and is_other:
                chat_message = "你看到了言哥和其他人在一起，你需要热情的打招呼并提醒他们给你介绍自己。"
            elif is_yan and not is_other:
                chat_message = "你看到了言哥过来，你需要热情的打招呼并主动交谈起来。"
            elif not is_yan and is_other:
                chat_message = "你看到一个陌生人，你需要热情的打招呼并提醒他给你介绍自己。"
            else:
                chat_message = "你好，当前场景无人"
            logger.info(f"当前场景识别到人物: {chat_message}")
     
            self.app._on_incoming_vision(chat_message)
    
            return 
        if  person_list.__len__() == 0 and self.person_list_holder.__len__() > 0:
            logger.info("当前场景无人，之前有人,进入休眠状态")
            chat_message = "当前场景无人，进入休眠状态"

            self.app._on_incoming_vision(chat_message)
    
            return 
        if self.person_list_holder.__len__() > 0 and person_list.__len__() > 0:

            if(len(person_list)==len(self.person_list_holder)):
                logger.info("人物列表无变化,表情、动作有变化")
                for new_person in person_list:
                    for old_person in self.person_list_holder:
                        if new_person.name==old_person.name:
                            logger.info(f"{new_person.name} 表情或动作发生了变化")
                            if new_person.emotion!=old_person.emotion:
                                logger.info(f" {new_person.name} 之前表情：{old_person.emotion}新表情为: {new_person.emotion}")
                                old_person.emotion=new_person.emotion
                                chat_message = f"检测到{new_person.name} 的表情发生了变化，之前是 {old_person.emotion}，现在是 {new_person.emotion}。针对表情变化，给出一些情感关怀"

                            
                            if  new_person.hand_gesture!=old_person.hand_gesture:
                                logger.info(f" {new_person.name} 之前手势：{old_person.hand_gesture}新手势为: {new_person.hand_gesture}")
                                old_person.hand_gesture=new_person.hand_gesture
                                chat_message=f"检测到{new_person.name} 有新手势动作 {new_person.hand_gesture}，请根据手势动作进行适当的回应"

                        else:
                            continue


            else:
                logger.info("人物列表有变化")
                # 比较person_list和self.person_list_holder的人物表格，找出以person.name新增的人物，并增加到self.person_list_holder中
                old_names = set(person.name for person in self.person_list_holder)
                new_names = set(person.name for person in person_list)
                added_names = new_names - old_names
                for person in person_list:
                    if person.name in added_names:
                        self.person_list_holder.append(person)
                        logger.info(f"新增人物: {person.name} 当前总人数= {len(self.person_list_holder)}")
                        if person.name == "yan2":
                            chat_message = "你看到了言哥过来，你需要热情的打招呼并主动交谈起来"
                        else:
                            chat_message = f"你看到一个陌生人，你需要热情的打招呼并提醒他给你介绍自己"
                

            self.app._on_incoming_vision(chat_message)
    
            return 
        else:
            logger.error("无预期的场景")  
            return  
        

        
     
    def start_person_analyze(self):
        if self.person_analyzer_thread is not None and self.person_analyzer_thread.is_alive():
            logger.warning("视觉分析线程已在运行")
            return
      
        
        self.person_analyzer_thread = threading.Thread(target=self._person_analyzer_loop, daemon=True)
        self.person_analyzer_thread.start()
        logger.info("视觉分析线程已启动")
      

    def stop_camera(self):
        """停止摄像头线程"""
        self.is_running = False
        if self.camera_thread is not None:
            self.camera_thread.join()  # 等待线程结束
            self.camera_thread = None
            logger.info("摄像头线程已停止")
            return {"status": "success", "message": "摄像头线程已停止"}
