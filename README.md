# AI_academi-EmbededAI
1. wifi setting
2. camera / ssh / vnc enable
3. 기타 실습 환경을 위한 설치

* 한글 깨짐
sudo apt-get install ibus
sudo apt-get install ibus-hangul
sudo apt-get install fonts-unfonts-core

* 실습 환경 세팅
pip3 install opencv-python==4.5.1.48
pip3 install numpy==1.20.2
sudo apt-get install libhdf5-dev -y
sudo apt-get install libhdf5-serial-dev -y
sudo apt-get install libatlas-base-dev -y
sudo apt-get install libjasper-dev -y
sudo apt-get install libqtgui4 -y
sudo apt-get install libqt4-test -y
pip3 install tensorflow==1.14.0
pip3 install keras==2.2.5
pip3 install h5py==2.10.0

======================
# 카메라 실행 여부 check 용 코드

import cv2

def main():
    camera = cv2.VideoCapture(-1)
    camera.set(3,640)
    camera.set(4,480)
    
    while(camera.isOpened()):
        keValue = cv2.waitKey(1)
        if keValue == ord('q'):
            break
        
        _, image = camera.read()
        image = cv2.flip(image, -1)
        cv2.imshow('raw', image)
        
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
