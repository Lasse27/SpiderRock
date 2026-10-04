#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_err.h"
#include "servo_motor.h"

#define SERVO_PIN GPIO_NUM_18
#define MIN_ANGLE 90
#define MAX_ANGLE 270
void app_main(void)
{
    setup_pwm(SERVO_PIN);
    set_servo_angle(MIN_ANGLE);
    vTaskDelay(pdMS_TO_TICKS(1000));
    while (1)
    {
        for (int i = MIN_ANGLE; i <= MAX_ANGLE; i++)
        {
            set_servo_angle(i);
            printf("%d degrees\n", i);
            vTaskDelay(pdMS_TO_TICKS(50));
        }

        for (int i = MAX_ANGLE; i >= MIN_ANGLE; i--)
        {
            set_servo_angle(i);
            printf("%d degrees\n", i);
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    }
}