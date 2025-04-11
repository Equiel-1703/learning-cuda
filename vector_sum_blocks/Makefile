HIP_FOLDER = hip
OBJS_FOLDER = obj
BIN_FOLDER = bin
COMMON_OBJS_FOLDER = ../common/obj

SRCS_CU = $(wildcard *.cu)
SRCS_CPP = $(wildcard *.cpp)
HIPS = $(patsubst %.cu,$(HIP_FOLDER)/%.hip,$(SRCS_CU)) $(patsubst %.cpp,$(HIP_FOLDER)/%.hip,$(SRCS_CPP))
COMMON_OBJS = $(wildcard $(COMMON_OBJS_FOLDER)/*.o)
BIN = $(patsubst %.cu,$(BIN_FOLDER)/%.out,$(SRCS_CU)) $(patsubst %.cpp,$(BIN_FOLDER)/%.out,$(SRCS_CPP))

HIPIFY = hipify-clang
CLANG_FLAGS = -std=c++17

COMPILER = hipcc

define HIPIFY_RULE
$(HIP_FOLDER)/%.hip: %.$(1) | $(HIP_FOLDER)
	@echo "Hipifying $$<"
	$$(HIPIFY) $$< -o $$@ -- $$(CLANG_FLAGS)
	@sed -i 's@\.\./common/include/@../common/hip/@g' $$@
endef

all: $(HIPS) $(OBJS) $(BIN)

$(BIN_FOLDER)/%.out: $(OBJS_FOLDER)/%.o | $(BIN_FOLDER)
	@echo "Linking $<"
	$(COMPILER) $(CLANG_FLAGS) -o $@ $< $(COMMON_OBJS)

$(OBJS_FOLDER)/%.o: $(HIP_FOLDER)/%.hip | $(OBJS_FOLDER)
	@echo "Compiling $<"
	$(COMPILER) $(CLANG_FLAGS) -c $< -o $@

$(eval $(call HIPIFY_RULE,cu))
$(eval $(call HIPIFY_RULE,cpp))

$(HIP_FOLDER):
	@echo "Creating hip folder"
	mkdir -p $(HIP_FOLDER)

$(OBJS_FOLDER):
	@echo "Creating obj folder"
	mkdir -p $(OBJS_FOLDER)

$(BIN_FOLDER):
	@echo "Creating bin folder"
	mkdir -p $(BIN_FOLDER)

clean:
	rm -f $(HIP_FOLDER)/*
	rm -f $(OBJS_FOLDER)/*
	rm -f $(BIN_FOLDER)/*