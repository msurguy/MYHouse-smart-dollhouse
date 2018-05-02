import os
import signal
import sys
from time import sleep, gmtime, strftime
import psmove
from neopixel import *
from gpiozero import Servo
import subprocess

VIDEO_PATH = "/home/pi/Desktop/video.mp4"

#import mxnet as mx
import numpy as np
from mxnet import nd, init, cpu, gluon
import time

def signal_handler(signal, frame):
        colorWipe(strip, Color(0,0,0))
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# LED strip configuration:
LED_COUNT      = 29      # Number of LED pixels.
LED_PIN        = 18      # GPIO pin connected to the pixels (18 uses PWM!).
#LED_PIN        = 10      # GPIO pin connected to the pixels (10 uses SPI /dev/spidev0.0).
LED_FREQ_HZ    = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA        = 10      # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255     # Set to 0 for darkest and 255 for brightest
LED_INVERT     = False   # True to invert the signal (when using NPN transistor level shift)
LED_CHANNEL    = 0       # set to '1' for GPIOs 13, 19, 41, 45 or 53
LED_STRIP      = ws.WS2811_STRIP_GRB   # Strip type and colour ordering

tv_is_playing = False

fan_servo_pin = 6
 
fan_servo_correction=0.45
fan_servo_maxPW=(2.0+fan_servo_correction)/1000
fan_servo_minPW=(1.0-fan_servo_correction)/1000


servo_left_pin = 4
servo_right_pin = 22 

servo_right_correction=0.45
servo_right_maxPW=(2.0+servo_right_correction)/1000
servo_right_minPW=(1.0-servo_right_correction)/1000

servo_left_correction=0.45
servo_left_maxPW=(2.0+servo_left_correction)/1000
servo_left_minPW=(1.0-servo_left_correction)/1000

servo_left = Servo(servo_left_pin, min_pulse_width=servo_left_minPW, max_pulse_width=servo_left_maxPW)
servo_right = Servo(servo_right_pin, min_pulse_width=servo_right_minPW, max_pulse_width=servo_right_maxPW)
fan_servo = Servo(fan_servo_pin, min_pulse_width=fan_servo_minPW, max_pulse_width=fan_servo_maxPW)

servo_left.value = 1
servo_right.value = -1

sleep(1)
servo_left.value = None
servo_right.value = None
fan_servo.value = None

shutters_opened = False
fan_on = False
fan_on_doublespeed = False

letter_m_is_on = False
letter_y_is_on = False

datapoints_iterator = 0 # iterator within datapoints
sample_iterator = 0 # iterator within the samples
datapoints_max = 40 # how many data points to collect for each sample
trigger_is_pressed = False

ctx = cpu()

# Define some variables
batch_size = 64
num_outputs = 7

# Define functions which animate LEDs in various ways.
def colorWipe(strip, color, wait_ms=50):
    """Wipe color across display a pixel at a time."""
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
        strip.show()
        sleep(wait_ms/1000.0)

# Define functions which animate LEDs in various ways.
def colorWipeRange(strip, color, range_min = 0, range_max = 1, wait_ms=50):
    """Wipe color across display a pixel at a time."""
    for i in range(range_min, range_max):
        strip.setPixelColor(i, color)
        strip.show()
        sleep(wait_ms/1000.0)


def wheel(pos, brightness = 255):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 85:
        color_with_brightness = brightness - pos * 3
        #if color_with_brightness < 0:
        #    color_with_brightness = 0
        return Color(pos * 3, color_with_brightness, 0)
    elif pos < 170:
        pos -= 85
        color_with_brightness = brightness - pos * 3
        #if color_with_brightness < 0:
        #    color_with_brightness = 0
        return Color(color_with_brightness, 0, pos * 3)
    else:
        pos -= 170
        color_with_brightness = brightness - pos * 3
        #if color_with_brightness < 0:
        #    color_with_brightness = 0
        return Color(0, pos * 3, color_with_brightness)

def rainbow(strip, wait_ms=20, iterations=1):
    """Draw rainbow that fades across all pixels at once."""
    for j in range(256*iterations):
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, wheel((i+j) & 255))
        strip.show()
        sleep(wait_ms/1000.0)

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


if psmove.count_connected() < 1:
    print('No controller connected')
    sleep (10)
    if psmove.count_connected() < 1:
        sys.exit(1)

move = psmove.PSMove()

# OrientationFusion_MadgwickIMU
#OrientationFusion_MadgwickIMU = _psmove.OrientationFusion_MadgwickIMU
#OrientationFusion_MadgwickMARG = _psmove.OrientationFusion_MadgwickMARG
#OrientationFusion_ComplementaryMARG = _psmove.OrientationFusion_ComplementaryMARG

# Important to enable getting orientation values as a calculated quaternion
move.enable_orientation(True)

if move.connection_type != psmove.Conn_Bluetooth:
    print('Please connect controller via Bluetooth')
    sys.exit(1)

assert move.has_calibration()


num_hidden = 128
net = gluon.nn.Sequential()
with net.name_scope():
    ###########################
    # Adding first hidden layer
    ###########################
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    ###########################
    # Adding dropout with rate .5 to the first hidden layer
    ###########################
    net.add(gluon.nn.Dropout(.5))

    ###########################
    # Adding first hidden layer
    ###########################
    net.add(gluon.nn.Dense(num_hidden, activation="relu"))
    ###########################
    # Adding dropout with rate .5 to the second hidden layer
    ###########################
    net.add(gluon.nn.Dropout(.5))

    ###########################
    # Adding the output layer
    ###########################
    net.add(gluon.nn.Dense(num_outputs))

net.collect_params().initialize(init.Xavier(magnitude=2.24), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

net.load_params('/home/pi/Desktop/ssd_300-working.params', ctx)
#net.load_params('/home/pi/Desktop/ssd_100-nicholas.params', ctx)
shutdown_count = 0

current_gesture_samples = []

result_dict = { 
    1:'tv-poke', 
    5:'fan-one-circle', 
    6:'fan-two-circles', 
    2:'shutter-right-left', 
    0:'no-gesture', 
    3:'letter-m', 
    4:'letter-y'
}

if __name__ == '__main__':
        # Process arguments
    # Create NeoPixel object with appropriate configuration.
        strip = Adafruit_NeoPixel(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL, LED_STRIP)
    # Intialize the library (must be called once before other functions).
        strip.begin()
        print ('Press Ctrl-C to quit.')
        while True:
            #rainbow(strip)
            trigger_value = move.get_trigger()

            if trigger_value > 20:
                trigger_is_pressed = True    

                if datapoints_iterator < datapoints_max:
                    datapoints_iterator = datapoints_iterator + 1
                    current_gesture_samples.extend([gx, gy, gz, ax, ay, az])

            if (trigger_value <= 10) and (trigger_is_pressed == True):
                trigger_is_pressed = False
                sample_iterator = sample_iterator + 1
                fill_empty_datapoints = datapoints_max - datapoints_iterator
                
                if (fill_empty_datapoints > 0):
                    for x in range(0, fill_empty_datapoints):
                        current_gesture_samples.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                current_gesture_array = np.asarray(current_gesture_samples, dtype=np.float64)
                print(current_gesture_array)

                result_num = int(net(nd.array(current_gesture_array.reshape(-1, 240))).argmax(axis=1).asscalar())

                print(result_dict[result_num])
                if (result_num == 1):
                    print('TV will be on')
                    if tv_is_playing == False:
                        os.system('sudo killall omxplayer.bin')
                        omxc = subprocess.Popen(['omxplayer', '-b', VIDEO_PATH])
                        tv_is_playing = True
                    else: 
                        os.system('sudo killall omxplayer.bin')
                        tv_is_playing = False

                    
                if (result_num == 2 ):
                    if shutters_opened == False:
                        servo_left.value = 0
                        servo_right.value = 0
                        sleep(0.5)
                        servo_left.value = None
                        servo_right.value = None
                        shutters_opened = True
                    else :
                        servo_left.value = 1
                        servo_right.value = -1
                        sleep(0.5)
                        servo_left.value = None
                        servo_right.value = None
                        shutters_opened = False
                if (result_num == 5 ):
                    if fan_on == False:
                        fan_servo.value = 0.8
                        sleep(1)
                        fan_on = True
                        fan_on_doublespeed = False
                    else:
                        fan_servo.value = None
                        fan_on = False
                        
                if (result_num == 6 ):
                    if fan_on_doublespeed == False:
                        fan_servo.value = 1
                        sleep(1)
                        fan_on_doublespeed = True
                        fan_on = False
                    else:
                        fan_servo.value = None
                        fan_on_doublespeed = False
                if (result_num == 3):
                    if letter_m_is_on == False :
                        letter_m_is_on = True
                        colorWipeRange(strip, Color(255,0,255), range_min = 0, range_max = 13)
                    else :
                        letter_m_is_on = False
                        colorWipeRange(strip, Color(0,0,0), range_min = 0, range_max = 13)

                if (result_num == 4):
                    if letter_y_is_on == False :
                        letter_y_is_on = True
                        colorWipeRange(strip, Color(255,200,0), range_min = 14, range_max = 28)
                    else :
                        letter_y_is_on = False
                        colorWipeRange(strip, Color(0,0,0), range_min = 14, range_max = 28)

                #result_dict = { 
                #    1:'tv-poke', 
                #    5:'fan-one-circle', 
                #    6:'fan-two-circles', 
                #    2:'shutter-right-left', 
                #    0:'no-gesture', 
                #    3:'letter-m', 
                #    4:'letter-y'
                #}
                datapoints_iterator = 0
                current_gesture_samples = []
                print ("Trigger is released, resetting gesture recognition")

            while move.poll():
                ax, ay, az = move.get_accelerometer_frame(psmove.Frame_SecondHalf)
                gx, gy, gz = move.get_gyroscope_frame(psmove.Frame_SecondHalf)
                qw, qx, qy, qz = move.get_orientation()
                brightness_scaled = int(translate(qx, -1, 1, 0, 255))
                color_scaled = wheel(int(translate(qz, -1, 1, 0, 255)), 255)
                #color_scaled_by_brightness = int(translate(qz, -1, 1, 0, 255))
                #print(brightness_scaled)
                #for i in range(strip.numPixels()):
                    #strip.setPixelColor(i, color_scaled)
                    #strip.setBrightness(brightness_scaled)
                    #strip.show()
                
                pressed, released = move.get_button_events()

                if (pressed == 96):
                    shutdown_count = shutdown_count + 1
                    if shutdown_count > 2:
                        colorWipe(strip, Color(0,0,0))
                        sleep (1)
                        print ("shutting down")
                        os.system("sudo poweroff")
                        #subprocess.call(["/sbin/shutdown", "-h", "now"])

                if pressed & psmove.Btn_TRIANGLE:
                    print('TRIANGLE pressed')
                    move.set_leds(0, 255, 255)
                    if fan_on == False:
                        fan_servo.value = -0.9
                        sleep(1)
                        fan_on = True
                    else :
                        fan_servo.value = None
                        fan_on = False
                    
                if pressed & psmove.Btn_SQUARE:
                    print('square pressed')
                    move.set_leds(0, 0, 255)
                    move.update_leds()
                    if shutters_opened == False:
                        servo_left.value = 0
                        servo_right.value = 0
                        sleep(0.5)
                        servo_left.value = None
                        servo_right.value = None
                        shutters_opened = True
                    else :
                        servo_left.value = 1
                        servo_right.value = -1
                        sleep(0.5)
                        servo_left.value = None
                        servo_right.value = None
                        shutters_opened = False
                
                if pressed & psmove.Btn_MOVE:
                    move.reset_orientation()
            
            sleep(0.045)

            
