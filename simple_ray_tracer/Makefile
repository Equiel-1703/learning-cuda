HIP_FOLDER = hip
BIN_FOLDER = bin

OBJS_FOLDER = obj
COMMON_OBJS_FOLDER = ../common/obj

INCLUDE_SRC_FOLDER = include_src
INCLUDE_FOLDER = include
INCLUDE_FLAG = -I$(INCLUDE_FOLDER)

SRCS_CU = $(wildcard *.cu)
SRCS_CPP = $(wildcard *.cpp)
INCLUDES_SRCS = $(wildcard $(INCLUDE_SRC_FOLDER)/*.cpp)

HIPS = $(patsubst %.cu,$(HIP_FOLDER)/%.hip,$(SRCS_CU)) $(patsubst %.cpp,$(HIP_FOLDER)/%.hip,$(SRCS_CPP))
HIPS += $(patsubst %.cpp,$(HIP_FOLDER)/%.hip,$(notdir $(INCLUDES_SRCS)))

OBJS = $(patsubst %.hip,$(OBJS_FOLDER)/%.o,$(notdir $(HIPS)))
COMMON_OBJS = $(wildcard $(COMMON_OBJS_FOLDER)/*.o)

BIN = $(patsubst %.cu,$(BIN_FOLDER)/%.out,$(SRCS_CU)) $(patsubst %.cpp,$(BIN_FOLDER)/%.out,$(SRCS_CPP))

HIPIFY = hipify-clang
CLANG_FLAGS = -std=c++17

COMPILER = hipcc

.PHONY: create_folders

all: create_folders $(HIPS) $(OBJS) $(BIN)

$(BIN_FOLDER)/%.out: $(OBJS)
	@echo "Linking $@"
	$(COMPILER) $(CLANG_FLAGS) -o $@ $(OBJS) $(COMMON_OBJS)

$(OBJS_FOLDER)/%.o: $(HIP_FOLDER)/%.hip
	@echo "Compiling $<"
	$(COMPILER) $(CLANG_FLAGS) -c $< -o $@ $(INCLUDE_FLAG)

define HIPIFY_RULE
$(HIP_FOLDER)/%.hip: %.$(1)
	@echo "Hipifying $$<"
	$$(HIPIFY) $$< -o $$@ $$(INCLUDE_FLAG) -- $$(CLANG_FLAGS)
	@sed -i 's@\.\./common/include/@../common/hip/@g' $$@
endef

$(eval $(call HIPIFY_RULE,cu))
$(eval $(call HIPIFY_RULE,cpp))

create_folders:
	@echo "Creating folders"
	@mkdir -p $(HIP_FOLDER) $(BIN_FOLDER) $(OBJS_FOLDER)

clean:
	rm -f $(HIP_FOLDER)/*
	rm -f $(OBJS_FOLDER)/*
	rm -f $(BIN_FOLDER)/*