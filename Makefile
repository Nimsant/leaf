GCC  ?= g++
NVCC := nvcc -ccbin $(GCC)
PY2  := python
PY3  := python3.7 

NVCCFLAGS   := -Xptxas="-v"
CCFLAGS     := -O3 -std=c++11
NVCCLDFLAGS :=
LDFLAGS     :=

EXTRA_NVCCFLAGS   ?=
EXTRA_NVCCLDFLAGS ?=
EXTRA_LDFLAGS     ?=
EXTRA_CCFLAGS     ?= -std=c++0x

NVCCLIBRARIES := -lcudart -lcufft -lpng
LIBRARIES := $(GCCLIBRARIES) $(NVCCLIBRARIES)

GENCODE_SM20  := -gencode arch=compute_20,code=sm_21 --maxrregcount 32
GENCODE_SM30  := -gencode arch=compute_30,code=sm_30
GENCODE_SM35  := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50  := -gencode arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM52  := -gencode arch=compute_52,code=\"sm_52,compute_52\" $(EXTRA_NVCCFLAGS)
GENCODE_SM60  := -gencode arch=compute_60,code=\"sm_60,compute_60\" $(EXTRA_NVCCFLAGS)
GENCODE_SM61  := -gencode arch=compute_61,code=\"sm_61,compute_61\" $(EXTRA_NVCCFLAGS)
GENCODE_SM70  := -gencode arch=compute_70,code=\"sm_70,compute_70\" $(EXTRA_NVCCFLAGS)
GENCODE_SM75  := -gencode arch=compute_75,code=\"sm_75,compute_75\" $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS   := --compiler-options="$(CCFLAGS) $(EXTRA_CCFLAGS)"
ALL_LDFLAGS   := --linker-options="$(LDFLAGS) $(EXTRA_LDFLAGS)"

where=$(shell hostname)
ifeq ($(where),Ion)
  GENCODE_FLAGS := $(GENCODE_SM52)
  NSM ?=13
  PY3 = /opt/python3.7/bin/python3.7
endif
ifeq ($(where),positron)
  GENCODE_FLAGS := $(GENCODE_SM61)
  NSM ?=15
  PY3 = /opt/python3.7/bin/python3.7
endif
ifeq ($(where),k60gpu)
  GENCODE_FLAGS := $(GENCODE_SM70)
  PY3  := /common/intel/intelpython3/bin/python3.5
  NSM ?=80
endif
ifeq ($(where),lev)
  GENCODE_FLAGS := $(GENCODE_SM75)
  NSM ?=30
endif
ifeq ($(where),D)
  GENCODE_FLAGS := $(GENCODE_SM35)
  NSM ?=14
  PY3 = /opt/python3.7/bin/python3.7
endif
ifeq ($(where),photon)
  GENCODE_FLAGS := $(GENCODE_SM61)
  NSM ?=10
  PY3 = /opt/python3.7/bin/python3.7
endif
ifeq ($(where),electron)
  GENCODE_FLAGS := $(GENCODE_SM52)
  NSM ?=22
endif
ifeq ($(where),tsubame)
  GENCODE_FLAGS := $(GENCODE_SM60)
  NSM ?=56
endif
ifeq ($(where),Zu)
  GENCODE_FLAGS := $(GENCODE_SM75)
  NSM ?=36
  PY3 = python3.7
  GCC = g++-7
  NVCC := nvcc -ccbin $(GCC)
endif

NAME = start_cuda

all: $(NAME)

obj_files := obj/main.o

obj/main.o: main.cu 
	$(EXEC) $(NVCC) $(CDEFS) $(CCFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(NAME): $(obj_files)
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) -o $@ $+ $(LIBRARIES)

run: $(NAME)
	$(EXEC) vglrun ./$(NAME) --zoom "$(zoom) $(zoom) $(zoom)" --Dmesh 0.5

clean:
	$(EXEC) rm -f (obj_files) main 

