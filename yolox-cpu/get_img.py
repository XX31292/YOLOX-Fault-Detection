import cv2 as cv#这个脚本实现了无人机摄像头实时采集图像并保存为图片文件的功能
import os#导入 OpenCV 和操作系统接口模块
os.makedirs("came_image/zhaopian", exist_ok=True) #用于保存图片的目录 came_image/zhaopian

image_number=30                #保存图片的数量
index = 0                        #定义初始图片编号0
#VideoCapture 是 OpenCV 库中用于从摄像头或视频文件中捕获图像的类
cap = cv.VideoCapture(0)           #0是电脑摄像头不加“”，或者加机械狗IP  不用运行
while cap.isOpened():
    tf, img = cap.read()#cap.read() 会从摄像头缓冲区内取出最新的一帧画面
    if not tf:#读取成功与否  t和f（True/False）
        break#如果读取失败 break 跳出整个 while 循环，不再继续
    cv.imshow("image", img)
    cv.imwrite(os.path.join("came_image/zhaopian", f"{index}.jpg"), img)
    index += 1#每保存一张图片，索引自增 1
    if index > image_number:#当 index > 30 时退出循环
        break
    print(f"已保存{index}张图片")#显示当前已经保存了多少张图片
    if cv.waitKey(1) & 0xFF == ord('q'): #当保存数量达到设定值或按 q 键时退出循环
        break

cap.release()#释放内存
cv.destroyAllWindows()#通过 cv.destroyAllWindows() 关闭所有 OpenCV 创建的窗口

