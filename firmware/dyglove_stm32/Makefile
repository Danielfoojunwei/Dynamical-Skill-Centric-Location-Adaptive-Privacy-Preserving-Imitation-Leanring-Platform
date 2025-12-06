# DYGlove STM32 Firmware Makefile
# Target: STM32F042K6T6 (ARM Cortex-M0, 32KB Flash, 6KB RAM)

######################################
# Target
######################################
TARGET = dyglove

######################################
# Building variables
######################################
DEBUG = 1
OPT = -Og

#######################################
# Paths
#######################################
BUILD_DIR = build

######################################
# Source files
######################################
C_SOURCES =  \
src/main.c \
src/stm32f0xx_hal.c \
src/stm32f0xx_hal_gpio.c \
src/stm32f0xx_hal_spi.c \
src/stm32f0xx_hal_i2c.c \
src/stm32f0xx_hal_uart.c \
src/stm32f0xx_hal_dma.c \
src/stm32f0xx_hal_tim.c \
src/stm32f0xx_hal_pcd.c \
src/usbd_core.c \
src/usbd_cdc.c \
src/ads1256.c \
src/dynamixel.c \
src/drv2605.c \
src/mpu6050.c \
src/system_stm32f0xx.c

# ASM sources
ASM_SOURCES =  \
startup_stm32f042x6.s

#######################################
# Binaries
#######################################
PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
AS = $(PREFIX)gcc -x assembler-with-cpp
CP = $(PREFIX)objcopy
SZ = $(PREFIX)size
HEX = $(CP) -O ihex
BIN = $(CP) -O binary -S

#######################################
# CFLAGS
#######################################
CPU = -mcpu=cortex-m0
FPU =
FLOAT-ABI =
MCU = $(CPU) -mthumb $(FPU) $(FLOAT-ABI)

# Macros
AS_DEFS = 
C_DEFS =  \
-DSTM32F042x6 \
-DUSE_HAL_DRIVER

# Includes
AS_INCLUDES = 
C_INCLUDES =  \
-Iinc \
-Idrivers/STM32F0xx_HAL_Driver/Inc \
-Idrivers/CMSIS/Device/ST/STM32F0xx/Include \
-Idrivers/CMSIS/Include

# Compile flags
ASFLAGS = $(MCU) $(AS_DEFS) $(AS_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections
CFLAGS = $(MCU) $(C_DEFS) $(C_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections

ifeq ($(DEBUG), 1)
CFLAGS += -g -gdwarf-2
endif

# Generate dependency information
CFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"

#######################################
# LDFLAGS
#######################################
LDSCRIPT = STM32F042K6Tx_FLASH.ld

LIBS = -lc -lm -lnosys 
LIBDIR = 
LDFLAGS = $(MCU) -specs=nano.specs -T$(LDSCRIPT) $(LIBDIR) $(LIBS) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref -Wl,--gc-sections

#######################################
# Build targets
#######################################
all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).hex $(BUILD_DIR)/$(TARGET).bin

# List of objects
OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))
vpath %.c $(sort $(dir $(C_SOURCES)))
OBJECTS += $(addprefix $(BUILD_DIR)/,$(notdir $(ASM_SOURCES:.s=.o)))
vpath %.s $(sort $(dir $(ASM_SOURCES)))

$(BUILD_DIR)/%.o: %.c Makefile | $(BUILD_DIR) 
	$(CC) -c $(CFLAGS) -Wa,-a,-ad,-alms=$(BUILD_DIR)/$(notdir $(<:.c=.lst)) $< -o $@

$(BUILD_DIR)/%.o: %.s Makefile | $(BUILD_DIR)
	$(AS) -c $(CFLAGS) $< -o $@

$(BUILD_DIR)/$(TARGET).elf: $(OBJECTS) Makefile
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	$(SZ) $@

$(BUILD_DIR)/%.hex: $(BUILD_DIR)/%.elf | $(BUILD_DIR)
	$(HEX) $< $@
	
$(BUILD_DIR)/%.bin: $(BUILD_DIR)/%.elf | $(BUILD_DIR)
	$(BIN) $< $@	
	
$(BUILD_DIR):
	mkdir $@		

#######################################
# Clean
#######################################
clean:
	-rm -fR $(BUILD_DIR)

#######################################
# Flash
#######################################
flash: $(BUILD_DIR)/$(TARGET).bin
	st-flash write $< 0x08000000

flash-dfu: $(BUILD_DIR)/$(TARGET).bin
	dfu-util -a 0 -D $< -s 0x08000000:leave

#######################################
# Dependencies
#######################################
-include $(wildcard $(BUILD_DIR)/*.d)
