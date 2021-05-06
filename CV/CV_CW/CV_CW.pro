TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += C:\openCV343\release\install\include

LIBS+= C:\openCV343\release\bin\libopencv_core343.dll
LIBS+= C:\openCV343\release\bin\libopencv_highgui343.dll
LIBS+= C:\openCV343\release\bin\libopencv_imgcodecs343.dll
LIBS+= C:\openCV343\release\bin\libopencv_imgproc343.dll
LIBS+= C:\openCV343\release\bin\libopencv_calib3d343.dll
LIBS+= C:\openCV343\release\bin\libopencv_photo343.dll
LIBS+= C:\openCV343\release\bin\libopencv_videoio343.dll
LIBS+= C:\openCV343\release\bin\libopencv_features2d343.dll
LIBS+= C:\openCV343\release\bin\libopencv_objdetect343.dll

SOURCES += \
        main.cpp

DISTFILES += \
    V1.avi \
    V2.avi \
    V3.avi
