#!/usr/bin/make -f

# My attempt at a makefile that will work reasonably for my Learning OpenCV3 exercises and examples.
# Goals are to be able to change it as little as possible for each project, have it create binaries
# in one step, and to be able to have it automatically run Google Test unit tests for some exercises.
# Since these are simple projects, am putting obj files in current dir and binaries in bin.


SRCS_C++ = main.cpp
TARGETNAME = exercises16
LIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lgregOpenCvHelp -lopencv_videoio -lopencv_video -lopencv_features2d

G++ = g++
SRCS = $(SRCS_C++)
OBJS = $(SRCS_C++:.cpp=.o)
BINDIR = bin/

all: 	$(OBJS)
	mkdir -p $(BINDIR)
	$(G++) -o $(BINDIR)$(TARGETNAME) $(OBJS) $(LIBS)


.PHONY: clean
clean:
	rm -f *.o $(BINDIR)$(TARGETNAME)

