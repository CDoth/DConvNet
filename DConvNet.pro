CONFIG -= qt

TEMPLATE = lib
CONFIG += staticlib

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    __dconvnet_global.cpp \
    layer.cpp \
    ocl_layer.cpp

HEADERS += \
    __dconvnet_global.h \
    layer.h \
    ocl_layer.h

DISTFILES += \
    _kernel_space.cl

# Default rules for deployment.
unix {
    target.path = $$[QT_INSTALL_PLUGINS]/generic
}
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += C:/DothProject/DSynapse
INCLUDEPATH += C:/DothProject/DTL

win32: LIBS += -LF:\NVIDIA_CUDA\CUDA\v11.3\lib\x64 -lOpenCL
INCLUDEPATH += F:\NVIDIA_CUDA\CUDA\v11.3\include
