################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (10.3-2021.10)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../X-CUBE-AI/App/app_x-cube-ai.c \
../X-CUBE-AI/App/saline_network.c \
../X-CUBE-AI/App/saline_network_data.c \
../X-CUBE-AI/App/saline_network_data_params.c 

OBJS += \
./X-CUBE-AI/App/app_x-cube-ai.o \
./X-CUBE-AI/App/saline_network.o \
./X-CUBE-AI/App/saline_network_data.o \
./X-CUBE-AI/App/saline_network_data_params.o 

C_DEPS += \
./X-CUBE-AI/App/app_x-cube-ai.d \
./X-CUBE-AI/App/saline_network.d \
./X-CUBE-AI/App/saline_network_data.d \
./X-CUBE-AI/App/saline_network_data_params.d 


# Each subdirectory must supply rules for building sources it contributes
X-CUBE-AI/App/%.o X-CUBE-AI/App/%.su: ../X-CUBE-AI/App/%.c X-CUBE-AI/App/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32L4R9xx -c -I../Core/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/CMSIS/Include -I../X-CUBE-AI/App -I../X-CUBE-AI -I../Middlewares/ST/AI/Inc -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-X-2d-CUBE-2d-AI-2f-App

clean-X-2d-CUBE-2d-AI-2f-App:
	-$(RM) ./X-CUBE-AI/App/app_x-cube-ai.d ./X-CUBE-AI/App/app_x-cube-ai.o ./X-CUBE-AI/App/app_x-cube-ai.su ./X-CUBE-AI/App/saline_network.d ./X-CUBE-AI/App/saline_network.o ./X-CUBE-AI/App/saline_network.su ./X-CUBE-AI/App/saline_network_data.d ./X-CUBE-AI/App/saline_network_data.o ./X-CUBE-AI/App/saline_network_data.su ./X-CUBE-AI/App/saline_network_data_params.d ./X-CUBE-AI/App/saline_network_data_params.o ./X-CUBE-AI/App/saline_network_data_params.su

.PHONY: clean-X-2d-CUBE-2d-AI-2f-App

