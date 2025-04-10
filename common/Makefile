SRCS_FOLDER = src
INCLUDES_FOLDER = include
OBJS_FOLDER = obj
HIP_FOLDER = hip

INCLUDES = $(wildcard $(INCLUDES_FOLDER)/*.hpp)
INCLUDE_FLAG = -I$(INCLUDES_FOLDER)

SRCS = $(wildcard $(SRCS_FOLDER)/*.cpp)
HIPS = $(patsubst $(SRCS_FOLDER)/%.cpp,$(HIP_FOLDER)/%.hip,$(SRCS))
HIPS += $(patsubst $(INCLUDES_FOLDER)/%.hpp,$(HIP_FOLDER)/%.hpp,$(INCLUDES))
OBJS = $(patsubst $(SRCS_FOLDER)/%.cpp,$(OBJS_FOLDER)/%.o,$(SRCS))

HIPIFY = hipify-clang
CLANG_FLAGS = -std=c++17

COMPILER = hipcc

# all: $(HIPS)
all: $(HIPS) $(OBJS)

$(OBJS_FOLDER)/%.o: $(HIP_FOLDER)/%.hip | $(OBJS_FOLDER)
	@echo "Compiling $<"
	$(COMPILER) $(CLANG_FLAGS) -c $< -o $@

$(HIP_FOLDER)/%.hip: $(SRCS_FOLDER)/%.cpp | $(HIP_FOLDER)
	@echo "Hipifying $<"
	$(HIPIFY) $< -o $@ $(INCLUDE_FLAG) -- $(CLANG_FLAGS)
	@sed -i 's/..\/include\/*//g' $@

$(HIP_FOLDER)/%.hpp: $(INCLUDES_FOLDER)/%.hpp | $(HIP_FOLDER)
	@echo "Hipifying include $<"
	$(HIPIFY) $< -o $@ $(INCLUDE_FLAG) -- $(CLANG_FLAGS)

$(HIP_FOLDER):
	@echo "Creating hip folder"
	mkdir -p $(HIP_FOLDER)

$(OBJS_FOLDER):
	@echo "Creating obj folder"
	mkdir -p $(OBJS_FOLDER)

clean:
	rm -f $(OBJS_FOLDER)/*
	rm -f $(HIP_FOLDER)/*