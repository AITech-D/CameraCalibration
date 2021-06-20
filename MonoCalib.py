#-*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import glob
import configparser

class MonoCameraCalibrator(object):
    def __init__(self,
                 data_root,
                 corner_size=(7, 11),
                 image_shape=(720, 1280),
                 square_size=50,
                 rectify_mode=0,
                 cali_file="MonoCalib_Para_720P.ini",
                 suffix="png"):
        super(MonoCameraCalibrator, self).__init__()
        self.data_root = data_root
        self.corner_h = corner_size[0]
        self.corner_w = corner_size[1]
        self.H, self.W = image_shape
        self.img_shape = (self.W, self.H)
        self.suffix = suffix
        self.cali_file = os.path.join(self.data_root, cali_file)
        self.square_size = square_size
        self.rectify_mode = rectify_mode

    def Run_Calibrator(self):

        if os.path.exists(self.cali_file):
            print("\n===> Read Calibration file from: {} ...".format(self.cali_file))
            self.read_cali_file(self.cali_file)

        else:
            print("\n===> Start Calibration ...")
            self.mono_calibrate()
            # self.evaluate_calibrate(rectify_mode=self.rectify_mode, prefix="png")
            self.save_cali_file(self.cali_file)


    def mono_calibrate(self):

        ''' ========= 一、角点检测 =========  '''
        # 寻找亚像素角点的参数，设置迭代终止条件,停止准则为最大循环次数30和最大误差容限0.001
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 设置世界坐标系下棋盘角点坐标 object points, 形式为 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.corner_h*self.corner_w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.corner_w, 0:self.corner_h].T.reshape(-1, 2)
        objp = objp * self.square_size  # Create real world coords. Use your metric.
        # 用arrays存储所有图片的object points 和 image points
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        # 用glob匹配文件夹/home/song/pic_1/right/下所有文件名含有“.jpg"的图片
        mono_images_path = os.path.join(self.data_root, "*.{}".format(self.suffix))
        image_path_list = glob.glob(mono_images_path)
        for i, fname in enumerate(image_path_list):
            img = cv2.imread(fname)
            print("{:05d}: {}, {}".format(i, fname, img.shape))

            h_, w_, _ = img.shape
            if h_ != self.H or w_ != self.W:
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 查找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, (self.corner_w, self.corner_h), None)
            if ret and (corners[0, 0, 0] < corners[-1, 0, 0]):
                print("*"*5+"order of {} is inverse!".format(i)+"*"*5)
                corners = np.flip(corners, axis=0).copy()

            # 如果找到了就添加 object points, image points
            if ret == True:
                objpoints.append(objp)
                # 在原角点的基础上计算亚像素角点
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                # 对角点连接画线加以展示
                cv2.drawChessboardCorners(img, (self.corner_w, self.corner_h), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(50)
        cv2.destroyAllWindows()

        self.objpoints = objpoints
        self.imgpoints = imgpoints


        ''' ========= 二、单目标定 =========  '''
        '''
        第一个参数objectPoints，为世界坐标系中的三维点。需要依据棋盘上单个黑白矩阵的大小，计算出（初始化）每一个内角点的世界坐标；
        第二个参数imagePoints，为每一个内角点对应的图像坐标点；
        第三个参数imageSize，为图像的像素尺寸大小，在计算相机的内参和畸变矩阵时需要使用到该参数；
        第四个参数cameraMatrix为相机的内参矩阵；
        第五个参数distCoeffs为畸变矩阵；
        第六个参数rvecs为旋转向量；
        第七个参数tvecs为位移向量；
        第八个参数flags为标定时所采用的算法。有如下几个参数：
            CV_CALIB_USE_INTRINSIC_GUESS：使用该参数时，在cameraMatrix矩阵中应该有fx,fy,u0,v0的估计值。否则的话，
                                          将初始化(u0,v0）图像的中心点，使用最小二乘估算出fx，fy。 
            CV_CALIB_FIX_PRINCIPAL_POINT：在进行优化时会固定光轴点。当CV_CALIB_USE_INTRINSIC_GUESS参数被设置，
                                          光轴点将保持在中心或者某个输入的值。 
            CV_CALIB_FIX_ASPECT_RATIO：固定fx/fy的比值，只将fy作为可变量，进行优化计算。
                                       当CV_CALIB_USE_INTRINSIC_GUESS没有被设置，fx和fy将会被忽略。
                                       只有fx/fy的比值在计算中会被用到。 
            CV_CALIB_ZERO_TANGENT_DIST：设定切向畸变参数（p1,p2）为零。 
            CV_CALIB_FIX_K1,…,CV_CALIB_FIX_K6：对应的径向畸变在优化中保持不变。 
            CV_CALIB_RATIONAL_MODEL：计算k4，k5，k6三个畸变参数。如果没有设置，则只计算其它5个畸变参数。
        第九个参数criteria是最优迭代终止条件设定。
        '''
        # flags = None
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        flags = None
        criteria = None
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints,
                                                                                    imgpoints,
                                                                                    gray.shape[::-1],
                                                                                    cameraMatrix=None,
                                                                                    distCoeffs=None)

        print("mtx: ", self.mtx)
        print("dist: ", self.dist)
        # print("rvecs: ", rvecs)
        # print("tvecs: ", tvecs)

        # 并且通过实验表明,distortion cofficients =(k_1,k_2,p_1,p_2,k_3)
        # 三个参数的时候由于k3所对应的非线性较为剧烈。估计的不好，容易产生极大的扭曲，所以k_3强制设置为0.0
        # self.dist[0,4]=0.0

        return self.ret, self.mtx, self.dist, self.rvecs, self.tvecs


    def rectify_image_list(self, mono_image_root, rectify_mode=0, prefix="png"):
        rectify_result_root = os.path.join(mono_image_root, "mono_rect")
        os.makedirs(rectify_result_root, exist_ok=True)

        # 对所有图片进行去畸变，有两种方法实现分别为： undistort()和remap()
        mono_images_path = os.path.join(self.data_root, "*.{}".format(self.suffix))
        image_path_list = glob.glob(mono_images_path)
        for i, fname in enumerate(image_path_list):
            mono_rect_path = os.path.join(rectify_result_root, "rect_{:05d}.{}".format(i, prefix))
            print("{:05d}:\n{}\n{}".format(i, fname, mono_rect_path))

            img = cv2.imread(fname)
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_CUBIC)

            image_rec = self.mono_rectify(img, mode=rectify_mode)
            cv2.imwrite(mono_rect_path, image_rec)




    def evaluate_calibrate(self, rectify_mode=0, prefix="png"):
        rectify_result_root = os.path.join(self.data_root, "calib_rect")
        os.makedirs(rectify_result_root, exist_ok=True)

        if self.ret:
            # 对所有图片进行去畸变，有两种方法实现分别为： undistort()和remap()
            mono_images_path = os.path.join(self.data_root, "*.{}".format(self.suffix))
            image_path_list = glob.glob(mono_images_path)
            for i, fname in enumerate(image_path_list):
                mono_rect_path = os.path.join(rectify_result_root, "rect_{:05d}.{}".format(i, prefix))
                print("{:05d}:\n{}\n{}".format(i, fname, mono_rect_path))

                img = cv2.imread(fname)
                img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
                # h, w = img.shape[:2]
                # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
                #
                # # # 使用 cv.undistort()进行畸变校正
                # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
                # # # 对图片有效区域进行剪裁
                # # # x, y, w, h = roi
                # # # dst = dst[y:y+h, x:x+w]
                # # cv2.imwrite('/home/song/pic_1/undistort/'+prefix, dst)
                #
                # #  使用 remap() 函数进行校正
                # # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
                # # dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                # # 对图片有效区域进行剪裁
                # # x, y, w, h = roi
                # # dst = dst[y:y + h, x:x + w]
                image_rec = self.mono_rectify(img, mode=rectify_mode)
                cv2.imwrite(mono_rect_path, image_rec)

            # 重投影误差计算
            mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error

            print("total error: ", mean_error / len(self.objpoints))


        else:
            raise ValueError("Mono calibrate failed! ")


    def mono_rectify(self, image, mode=0):
        if mode == 0:
            # 矫正单目图像：直接使用计算的相机内参数
            image_rec = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        elif mode == 1:
            # 矫正单目图像：使用计算的相机内参数计算新的内参矩阵
            '''
            1、默认情况下，我们通常不会求取新的CameraMatrix，这样代码中会默认使用标定得到的Cameralatrix.
               而这个摄像机矩阵是在理想情况下没有考虑畸变得到的，所以并不准确，重要的是fx和fy的值会比考虑畸变情况下的偏大，
               会损失很多有效像素。我们可以通过这个函数getoptimaNewCameramatrix()求取一个新的摄像机内参矩阵。
            2、cv2.getoptimalNewCameraMatrix()。如果参数alpha=0，它返回含有最小不需要像素的非扭曲图像，
               所以它可能移除一些图像角点。如果alpha=1,所有像素都返回。还会返回一个 Ror图像，我们可以用来对结果进行裁剪。
            '''
            # 使用 remap() 函数进行校正
            w, h = self.W, self.H
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 0, (w, h))
            image_rec = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
            # 对图片有效区域进行剪裁
            # x, y, w, h = roi
            # image_rec = image_rec[y:y + h, x:x + w]

        elif mode == 2:
            # 使用 remap() 函数进行校正
            w, h = self.W, self.H
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, newcameramtx, (w, h), 5)
            image_rec = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            # 对图片有效区域进行剪裁
            x, y, w, h = roi
            image_rec = image_rec[y:y + h, x:x + w]
        else:
            pass


        return image_rec


    def read_cali_file(self, cali_file):
        con = configparser.ConfigParser()
        con.read(cali_file, encoding='utf-8')
        sections = con.sections()
        # print(sections.items)
        calib = con.items('Calib')
        rectify = con.items('Rectify')
        calib = dict(calib)
        rectify = dict(rectify)

        self.u0 = calib['u0']
        self.v0 = calib['v0']
        self.fx = calib['fx']
        self.fy = calib['fy']

        self.mtx = np.array([[rectify['mtx_0'], rectify['mtx_1'], rectify['mtx_2']],
                             [rectify['mtx_3'], rectify['mtx_4'], rectify['mtx_5']],
                             [rectify['mtx_6'], rectify['mtx_7'], rectify['mtx_8']]]).astype('float32')

        self.dist = np.array([[rectify['dist_k1'],
                               rectify['dist_k2'],
                               rectify['dist_p1'],
                               rectify['dist_p2'],
                               rectify['dist_k3']]]).astype('float32')


    def save_cali_file(self, cali_file):
        self.u0 = self.mtx[0, 2]
        self.v0 = self.mtx[1, 2]
        self.fx = self.mtx[0, 0]
        self.fy = self.mtx[1, 1]
        self.Calib = {"u0": self.u0,
                      "v0": self.v0,
                      "fx": self.fx,
                      "fy": self.fy,
                      }
        # 相机内参矩阵
        # fx s u0
        # 0 fy v0
        # 0  0  1
        Rectify = {}
        for i in range(3):
            for j in range(3):
                Rectify["mtx_{}".format(i*3+j)] = self.mtx[i, j]

        # 相机畸变参数 distortion cofficients = (k_1, k_2, p_1, p_2, k_3)
        Rectify["dist_k1"] = self.dist[0, 0]
        Rectify["dist_k2"] = self.dist[0, 1]
        Rectify["dist_p1"] = self.dist[0, 2]
        Rectify["dist_p2"] = self.dist[0, 3]
        Rectify["dist_k3"] = self.dist[0, 4]

        self.Rectify = Rectify
        config = configparser.ConfigParser()
        config["Calib"] = self.Calib
        config["Rectify"] = self.Rectify
        with open(cali_file, 'w') as configfile:
            config.write(configfile)



def main():
    data_root = "./outputs/Data/Trinocular/Calib/mono"
    cameraCalibrator = MonoCameraCalibrator(data_root=data_root,
                                            corner_size=(7, 11),
                                            image_shape=(720, 1280),
                                            square_size=50,
                                            rectify_mode=1,
                                            cali_file="MonoCalib_Para_720P.ini",
                                            suffix="bmp")
    cameraCalibrator.Run_Calibrator()
    cameraCalibrator.rectify_image_list(mono_image_root=data_root, rectify_mode=0)



if __name__ == '__main__':
    main()








