#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import glob
import cv2
import argparse
import configparser


class StereoCameraCalibrator(object):
    def __init__(self,
                 data_root,
                 imgL_root=None,
                 imgR_root=None,
                 corner_size=(7, 11),
                 suffix="png",
                 image_shape=(720, 1280),
                 camera_param_file="StereoParameters_720P.ini"):
        super(StereoCameraCalibrator, self).__init__()
        self.data_root = data_root
        self.corner_h, self.corner_w = corner_size
        self.H, self.W = image_shape
        self.square_size = 50
        self.img_shape = (self.W, self.H)
        self.cali_file = os.path.join(data_root, camera_param_file)
        self.suffix = suffix

        if imgL_root == None and imgR_root == None:
            self.imgL_root = os.path.join(self.data_root, 'imgL/*.{}'.format(suffix))
            self.imgR_root = os.path.join(self.data_root, 'imgR/*.{}'.format(suffix))
        else:
            self.imgL_root = os.path.join(imgL_root, '*.{}'.format(suffix))
            self.imgR_root = os.path.join(imgR_root, '*.{}'.format(suffix))

    '''将左右图分开'''

    def split_imgLR(self, dir_name="imgLR", suffix="bmp"):
        imgLR_path_list = glob.glob(os.path.join(self.data_root, dir_name, "*.{}".format(suffix)))
        imgLR_path_list.sort()
        print(os.path.join(self.data_root, dir_name, "*.{}".format(suffix)))

        imgL_root = os.path.join(self.data_root, "imgL")
        imgR_root = os.path.join(self.data_root, "imgR")

        if not os.path.exists(imgL_root):
            os.makedirs(imgL_root)
            os.makedirs(imgR_root)

        for i, imgLR_path in enumerate(imgLR_path_list):
            imgLR = cv2.imread(imgLR_path)
            imgL, imgR = imgLR[:, :1920], imgLR[:, 1920:]

            imgL_path = os.path.join(imgL_root, "imgL_{:05d}.png".format(i))
            imgR_path = os.path.join(imgR_root, "imgR_{:05d}.png".format(i))

            cv2.imwrite(imgL_path, imgL)
            cv2.imwrite(imgR_path, imgR)
            print("{:05d}: {}".format(i, imgL_path))
            print("{:05d}: {}\n".format(i, imgR_path))

    '''立体标定'''
    def Run_Calibrator(self, alpha=0):
        if os.path.exists(self.cali_file):
            self.read_cali_file(self.cali_file)

        else:
            # calibration.
            self.read_images()
            self.camera_model = self.stereo_calibrate(self.img_shape, alpha=alpha)
            self.save_cali_file(self.cali_file)

    '''读取左右图角点'''

    def read_images(self):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.corner_h * self.corner_w, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.corner_w, 0:self.corner_h].T.reshape(-1, 2)
        self.objp *= self.square_size
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.

        images_left = glob.glob(self.imgL_root)
        images_right = glob.glob(self.imgR_root)
        images_left.sort()
        images_right.sort()
        if len(images_left) == len(images_right):
            for i, fname in enumerate(images_left):
                print("\nRead image {}: \n{}\n{}".format(i, images_left[i], images_right[i]))
                img_l = cv2.imread(images_left[i])
                img_r = cv2.imread(images_right[i])
                img_l = cv2.resize(img_l, self.img_shape)
                img_r = cv2.resize(img_r, self.img_shape)

                gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret_l, corners_l = cv2.findChessboardCorners(gray_l, (self.corner_w, self.corner_h), None)
                ret_r, corners_r = cv2.findChessboardCorners(gray_r, (self.corner_w, self.corner_h), None)

                # If found, add object points, image points (after refining them)
                self.objpoints.append(self.objp)
                if ret_l is True:
                    if (corners_l[0, 0, 0] < corners_l[-1, 0, 0]):
                        print("*" * 5 + "order of {} is inverse!".format(i) + "*" * 5)
                        corners_l = np.flip(corners_l, axis=0).copy()

                    rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                    self.imgpoints_l.append(corners_l)

                    # Draw and display the corners
                    ret_l = cv2.drawChessboardCorners(img_l, (self.corner_w, self.corner_h), corners_l, ret_l)
                    cv2.imshow("imgL", img_l)
                    cv2.waitKey(50)

                if ret_r is True:
                    if (corners_r[0, 0, 0] < corners_r[-1, 0, 0]):
                        print("*" * 5 + "order of {} is inverse!".format(i) + "*" * 5)
                        corners_r = np.flip(corners_r, axis=0).copy()

                    rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                    self.imgpoints_r.append(corners_r)

                    # Draw and display the corners
                    ret_r = cv2.drawChessboardCorners(img_r, (self.corner_w, self.corner_h), corners_r, ret_r)
                    cv2.imshow("imgR", img_r)
                    cv2.waitKey(50)
                # self.img_shape = gray_l.shape[::-1]
            cv2.destroyAllWindows()
        else:
            raise ValueError("left image and right image must be same!")

    def stereo_calibrate(self, img_shape, alpha=0):
        flag1 = 0
        flag1 |= cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
        rt, self.M1, self.D1, self.r1, self.t1 = cv2.calibrateCamera(self.objpoints,
                                                                     self.imgpoints_l,
                                                                     self.img_shape,
                                                                     None,
                                                                     None,
                                                                     flags=flag1)
        rt, self.M2, self.D2, self.r2, self.t2 = cv2.calibrateCamera(self.objpoints,
                                                                     self.imgpoints_r,
                                                                     self.img_shape,
                                                                     None,
                                                                     None,
                                                                     flags=flag1)

        # flag2 |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flag2 |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flag2 |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flag2 |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        '''
        1.objectPoints-  vector<point3f> 型的数据结构，存储标疋用息住世子
        2.imagePoints1- vector<vector<point2f>>型的数据结构，存储标定角点在第一个摄像机下的投影后的亚像素坐标;
        3.imagePoints2- vector<vector<point2f>>型的数据结构，存储标定角点在第二个摄像机下的投影后的亚像素坐标;
        4.cameraMatrix1 输入/出的第个摄像机的相机矩阵。如果CALIB_USE_INTRINSIC_GUESS, CV CALIB FIX ASPECT RATIO
         CV CALIB FIX INTRINSICCVCALIB FIXFOCALLENGTH其中的一个或多个标志被没置，该摄像机矩阵的一些或全部参数需要被初始化;
        5.distcoeffs1-第一个摄像机的辅入/输出型骑空问量，根据矫正模型的不同，输出向量长度由标志决定;
        6.cameraMatrix2-输入/输出型的第二个摄像机的相机矩阵。参数意义同第一个相机矩阵相似;
        7.distCoeffs2-第一个摄像机的输入/输出型变向量。根据矫正模型的不同，输出向量长度由标志决定;
        8.imageSize-图像的大小;
        9.R-输出型，第一和第二个摄像机之间的旋转矩阵;
        10.T-输出型，第一和第二个摄像机之间的平移矩阵;
        11.E-输出型，基本矩阵;
        12.F-输出型，基础矩阵;
         flag:标定时的一些选项:
         CALIS USEINTRINSICGUESS:使用该参数时，在cameraMatrix矩阵中应该有fx,fy,ue,ve的估计值。否则的话，将初始化(ue,ve)
        图像的中心点，使用最小二乘估算出fx.fy。
         CALIB FIXLPRINCIPAL_POINT:在进行优化时会固定光轴点。当CVCALIBUSE_INTRINSICGUESS参数被设置，光轴点将保持在中心或者某个输入的值。 CALIB FIX ASPECT RATI0:固定fx/fy的比值，只将fy作为可变量，进行优化计算。当CV CALIB USE_INTRINSIC GUESS没有被设置，
         fx和fy将会被忽略。只有fx/fy的比值在计算中会被用到。
         CALIB ZERO_TANGENT_DIST:设定切向畸变参数(p1,p2)为零。
         CALIB FIXK1,...,CALIB FIX_K6:对应的径向畸变在优化中保持不变
         CALIB RATIONAL MODEL:计算k4，k5，k6三个畸变参数。如果没有设置，则只计算其它5个畸变参数。
        '''
        flag2 = 0
        flag2 |= cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, self.K1, self.D1, self.K2, self.D2, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.objpoints,
            self.imgpoints_l,
            self.imgpoints_r,
            self.M1,
            self.D1,
            self.M2,
            self.D2,
            img_shape,
            criteria=stereocalib_criteria,
            flags=flag2)

        # 进行立体更正
        '''
        cameraMatrix1-第一个摄像机的摄像机矩阵 distCoeffs1-第一个摄像机的畸变向量
        cameraMatrix2-第二个摄像机的摄像机矩阵 distCoeffs1-第二个摄像机的畸变向量
        imageSize-图像大小
        R- stereoCalibrate()求得的R矩阵
        T- stereoCalibrate()求得的T矩阵
        R1-输出矩阵，第一个摄像机的校正变换矩阵(旋转变换) 
        R2-输出矩阵，第二个摄像机的校正变换矩阵(旋转矩阵) 
        P1-输出矩阵，第一个摄像机在新坐标系下的投影矩阵 
        P2-输出矩阵，第二个摄像机在想坐标系下的投影矩阵 
        Q-4*4的深度差异映射矩阵
        flags-可选的标志有两种零或者CV_CALIB_ZERO_DISPARITY，如果设置CV_CALIB_ZERO_DISPARITY 的话，
        该函数会让两幅校正后的图像的主点有相同的像素坐标。否则该函数会水平或垂直的移动图像，以使得其有用的范围最大 
        alpha-拉伸参数。如果设置为负或忽略，将不进行拉伸。
            如果设置为0，那么校正后图像只有有效的部分会被显示(没有黑色的部分)
            如果设置为1，那么就会显示整个图像。设置为0~1之间的某个值，其效果也居于两者之间。 
        newImageSize-校正后的图像分辨率，默认为原分辨率大小。
        validPixR0I1-可选的输出参数，Rect型数据。其内部的所有像素都有效 
        validPixR012-可选的输出参数，Rect型数据。其内部的所有像素都有效
        '''
        self.R1, self.R2, self.P1, self.P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
            self.K1, self.D1,
            self.K2, self.D2,
            self.img_shape,
            self.R,
            self.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=alpha)

    def get_coor(self, shape, axis=0):
        h, w = shape
        if axis == 0:
            coor = np.reshape(np.array(range(w)), [1, w])
            coor = coor.repeat(h, axis=axis)
        else:
            coor = np.reshape(np.array(range(h)), [h, 1])
            coor = coor.repeat(w, axis=axis)
        return coor.reshape(h, w)

    def undistort(self, image, K, D, R, P, interpolation=cv2.INTER_LINEAR):
        h, w = image.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(K, D, R, P, (w, h), cv2.CV_32FC1)
        image_Rectify = cv2.remap(image, mapx, mapy, interpolation)

        return image_Rectify

    def stereo_Rectify(self, imgL, imgR, shape=(720, 1280)):
        imgL = cv2.resize(imgL, (shape[1], shape[0]))
        imgL_rec = self.undistort(imgL, self.K1, self.D1, self.R1, self.P1)

        imgR = cv2.resize(imgR, (shape[1], shape[0]))
        imgR_rec = self.undistort(imgR, self.K2, self.D2, self.R2, self.P2)
        mapx = self.get_coor(shape=shape, axis=0)
        mapy = self.get_coor(shape=shape, axis=1)
        mapx = mapx - (float(self.Calib["u0l"]) - float(self.Calib["u0r"]))

        mapx = mapx.astype("float32")
        mapy = mapy.astype("float32")
        imgR_rec = cv2.remap(imgR_rec, mapx, mapy, cv2.INTER_LINEAR)

        # # 计算更正map
        # self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(self.K1,
        #                                                              self.D1,
        #                                                              self.R1,
        #                                                              self.P1,
        #                                                              self.img_shape,
        #                                                              cv2.CV_32FC1)
        # self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(self.K2,
        #                                                                self.D2,
        #                                                                self.R2,
        #                                                                self.P2,
        #                                                                self.img_shape,
        #                                                                cv2.CV_32FC1)
        #
        # imgL_Rectify = cv2.remap(imgL, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        # imgR_Rectify = cv2.remap(imgR, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

        return imgL_rec, imgR_rec

    def rectify_image_list(self, stereo_image_root, shape=(720, 1280), prefix="png"):

        # 对所有图片进行去畸变，有两种方法实现分别为： undistort()和remap()
        imgL_path_root = os.path.join(self.data_root, "imgL/*.{}".format(self.suffix))
        imgL_path_list = glob.glob(imgL_path_root)
        imgR_path_root = os.path.join(self.data_root, "imgR/*.{}".format(self.suffix))
        imgR_path_list = glob.glob(imgR_path_root)
        imgL_path_list.sort()
        imgR_path_list.sort()
        if len(imgL_path_list) == len(imgR_path_list):
            rec_imgL_root = os.path.join(stereo_image_root, "imgL_rec")
            rec_imgR_root = os.path.join(stereo_image_root, "imgR_rec")
            os.makedirs(rec_imgL_root, exist_ok=True)
            os.makedirs(rec_imgR_root, exist_ok=True)

            for i in range(len(imgL_path_list)):
                imgL_path = imgL_path_list[i]
                imgR_path = imgR_path_list[i]
                imgL_rec_path = os.path.join(rec_imgL_root, "imgL_rec_{:05d}.{}".format(i, prefix))
                imgR_rec_path = os.path.join(rec_imgR_root, "imgR_rec_{:05d}.{}".format(i, prefix))
                print("{:05d}:\n{}\n{}".format(i, imgL_path, imgL_rec_path))
                print("{:05d}:\n{}\n{}".format(i, imgR_path, imgR_rec_path))

                imgL = cv2.imread(imgL_path)
                imgR = cv2.imread(imgR_path)
                imgL = cv2.resize(imgL, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
                imgR = cv2.resize(imgR, (self.W, self.H), interpolation=cv2.INTER_CUBIC)

                imgL_rec, imgR_rec = self.stereo_Rectify(imgL, imgR, shape=shape)
                cv2.imwrite(imgL_rec_path, imgL_rec)
                cv2.imwrite(imgR_rec_path, imgR_rec)

    def read_cali_file(self, cali_file):
        con = configparser.RawConfigParser()
        con.optionxform = lambda option: option
        con.read(cali_file, encoding='utf-8')
        sections = con.sections()
        # print(sections.items)
        calib = con.items('Calib')
        rectify = con.items('Rectify')
        calib = dict(calib)
        rectify = dict(rectify)

        distort_left = np.array([rectify['KC0'], rectify['KC1'], rectify['KC2'], rectify['KC3'], 0.0]).astype('float32')
        distort_right = np.array(
            [rectify['KC_RIGHT0'], rectify['KC_RIGHT1'], rectify['KC_RIGHT2'], rectify['KC_RIGHT3'], 0.0]).astype(
            'float32')

        R_left = np.array([rectify["R{}".format(i)] for i in range(9)], dtype="float32").reshape(3, 3)
        R_right = np.array([rectify["R_RIGHT{}".format(i)] for i in range(9)], dtype="float32").reshape(3, 3)

        K_left_old = np.array([[rectify['FC0'], 0.0, rectify['CC0']],
                               [0.0, rectify['FC1'], rectify['CC1']],
                               [0.0, 0.0, 1.0]]).astype('float32')
        K_left_new = np.array([[calib['focus'], 0.0, calib['u0l']],
                               [0.0, calib['focus'], calib['v0']],
                               [0.0, 0.0, 1.0]]).astype('float32')

        K_right_old = np.array([[rectify['FC_RIGHT0'], 0.0, rectify['CC_RIGHT0']],
                                [0.0, rectify['FC_RIGHT1'], rectify['CC_RIGHT1']],
                                [0.0, 0.0, 1.0]]).astype('float32')
        K_right_new = np.array([[calib['focus'], 0.0, calib['u0r']],
                                [0.0, calib['focus'], calib['v0']],
                                [0.0, 0.0, 1.0]]).astype('float32')

        KK_mtx_left = np.array([rectify["KK_inv{}".format(i)] for i in range(9)], dtype="float32").reshape(3, 3)
        KK_mtx_right = np.array([rectify["KK_RIGHT{}".format(i)] for i in range(9)], dtype="float32").reshape(3, 3)

        Rectify = {
            "K1": K_left_old,
            "D1": distort_left,
            "R1": R_left,
            "P1": K_left_new,
            "K2": K_right_old,
            "D2": distort_right,
            "R2": R_right,
            "P2": K_right_new,
            "mtx1": KK_mtx_left,
            "mtx2": KK_mtx_right,
        }

        Calib = {
            "u0l": float(calib["u0l"]),
            "u0r": float(calib["u0r"]),
            "v0": float(calib["v0"]),
            "bline": float(calib["bline"]),
            "focus": float(calib["focus"]),
        }

        self.Calib = Calib
        self.Rectify = Rectify

        self.u0l = Calib['u0l']
        self.u0r = Calib['u0r']
        self.v0 = Calib['v0']
        self.baseline = Calib['bline']
        self.focus = Calib['focus']
        self.K1 = Rectify['K1']
        self.D1 = Rectify['D1']
        self.R1 = Rectify['R1']
        self.P1 = Rectify['P1']
        self.K2 = Rectify['K2']
        self.D2 = Rectify['D2']
        self.R2 = Rectify['R2']
        self.P2 = Rectify['P2']
        self.mtx1 = Rectify['mtx1']
        self.mtx2 = Rectify['mtx2']




    def save_cali_file(self, cali_file):
        self.u0l = self.P1[0, 2]
        self.u0r = self.P2[0, 2]
        self.v0 = self.P1[1, 2]
        self.baseline = 1.0 / self.Q[3, 2]
        self.focus = self.P1[0, 0]
        self.Calib = {
            "u0l": self.u0l,
            "u0r": self.u0r,
            "v0": self.v0,
            "bline": self.baseline,
            "focus": self.focus,
        }

        Rectify = {}
        R_left = np.reshape(self.R1, -1)
        for i in range(R_left.shape[0]):
            Rectify['R{}'.format(i)] = R_left[i]
        K_left_old = self.M1
        Rectify['FC0'] = K_left_old[0, 0]
        Rectify['FC1'] = K_left_old[1, 1]
        Rectify['CC0'] = K_left_old[0, 2]
        Rectify['CC1'] = K_left_old[1, 2]

        distort_left = self.D1
        for j in range(8):
            if j < 5:
                Rectify["KC{}".format(j)] = distort_left[0, j]
            else:
                Rectify["KC{}".format(j)] = 0.0

        for i in range(3):
            for j in range(3):
                Rectify["KK_inv{}".format(i * 3 + j)] = self.K1[i, j]

        R_right = np.reshape(self.R2, -1)
        for i in range(R_right.shape[0]):
            Rectify['R_RIGHT{}'.format(i)] = R_left[i]
        K_right_old = self.M2
        Rectify['FC_RIGHT0'] = K_right_old[0, 0]
        Rectify['FC_RIGHT1'] = K_right_old[1, 1]
        Rectify['CC_RIGHT0'] = K_right_old[0, 2]
        Rectify['CC_RIGHT1'] = K_right_old[1, 2]

        distort_right = self.D2
        for j in range(8):
            if j < 5:
                Rectify["KC_RIGHT{}".format(j)] = distort_left[0, j]
            else:
                Rectify["KC_RIGHT{}".format(j)] = 0.0

        for i in range(3):
            for j in range(3):
                Rectify["KK_RIGHT{}".format(i * 3 + j)] = self.K2[i, j]

        Rectify["KC4"] = 0.000000
        Rectify["KC2_0"] = 0.000000
        Rectify["KC2_1"] = 0.000000
        Rectify["KC2_2"] = 0.000000
        Rectify["KC2_RIGHT0"] = 0.000000
        Rectify["KC2_RIGHT1"] = 0.000000
        Rectify["KC2_RIGHT2"] = 0.000000

        self.Rectify = Rectify

        config = configparser.RawConfigParser()
        config.optionxform = lambda option: option
        config["Calib"] = self.Calib
        config["Rectify"] = self.Rectify
        with open(cali_file, 'w') as configfile:
            config.write(configfile)

    def rectify_video(self, video_path: str):
        self.load_params()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Unable to open video.")
            return False
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        out_format = video_path.split('.')[-1]
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(filename='out.' + out_format, fourcc=0x00000021, fps=fps, frameSize=self.image_size)
        cv2.namedWindow("origin", cv2.WINDOW_NORMAL)
        cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(frame_count):
            ret, img = cap.read()
            if ret:
                img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
                cv2.imshow("origin", img)
                dst = self.rectify_image(img)
                cv2.imshow("dst", dst)
                out.write(dst)
                cv2.waitKey(1)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return True


if __name__ == '__main__':
    img_H, img_W = 720, 1280  # 720, 1280, 1080, 1920
    data_root = "./outputs/Data/Trinocular/Calib"
    StereoCalibrator = StereoCameraCalibrator(data_root=data_root,
                                              imgL_root=None,
                                              imgR_root=None,
                                              corner_size=(7, 11),
                                              suffix="png",
                                              image_shape=(img_H, img_W),
                                              camera_param_file="StereoParameters_{}P.ini".format(img_H))

    StereoCalibrator.Run_Calibrator(alpha=1)
    # StereoCalibrator.rectify_image_list(stereo_image_root=data_root, shape=(img_H, img_W), prefix="png")

    images_right = glob.glob(os.path.join(data_root, 'imgL/*.png'))
    images_left = glob.glob(os.path.join(data_root, 'imgR/*.png'))
    images_left.sort()
    images_right.sort()
    imgL, imgR = cv2.imread(images_left[0]), cv2.imread(images_right[0])

    imgL_Rectify, imgR_Rectify = StereoCalibrator.stereo_Rectify(imgL, imgR, shape=(img_H, img_W))
    print("imgL_Rectify: ", imgL_Rectify.shape)
    print("imgR_Rectify: ", imgR_Rectify.shape)

    # 画圆，圆心为：(160, 160)，半径为：60，颜色为：point_color，实心线
    cv2.circle(imgL_Rectify, (int(img_W/2), int(img_H/2)), 3, (0, 0, 255), 0)
    cv2.circle(imgR_Rectify, (int(img_W/2), int(img_H/2)), 3, (0, 0, 255), 0)

    cv2.imshow("imgL_Rectify", imgL_Rectify)
    cv2.imshow("imgR_Rectify", imgR_Rectify)
    cv2.waitKey(0)
