HIP_FOLDER = hip
OBJS_FOLDER = obj
BIN_FOLDER = bin
COMMON_OBJS_FOLDER = ../common/obj

SRCS = $(wildcard *.cu)
HIPS = $(patsubst %.cu,$(HIP_FOLDER)/%.hip,$(SRCS))
COMMON_OBJS = $(wildcard $(COMMON_OBJS_FOLDER)/*.o)
BIN = $(patsubst %.cu,$(BIN_FOLDER)/%.out,$(SRCS))

HIPIFY = hipify-clang
CLANG_FLAGS = -std=c++17

COMPILER = hipcc

# all: $(HIPS)
all: $(HIPS) $(OBJS) $(BIN)

$(BIN_FOLDER)/%.out: $(OBJS_FOLDER)/%.o | $(BIN_FOLDER)
	@echo "Linking $<"
	$(COMPILER) $(CLANG_FLAGS) -o $@ $< $(COMMON_OBJS)

$(OBJS_FOLDER)/%.o: $(HIP_FOLDER)/%.hip | $(OBJS_FOLDER)
	@echo "Compiling $<"
	$(COMPILER) $(CLANG_FLAGS) -c $< -o $@

$(HIP_FOLDER)/%.hip: %.cu | $(HIP_FOLDER)
	@echo "Hipifying $<"
	$(HIPIFY) $< -o $@ -- $(CLANG_FLAGS)
	@sed -i 's/..\/common\/include\/*/..\/common\/hip\//g' $@

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