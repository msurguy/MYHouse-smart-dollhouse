#from websocket_server import WebsocketServer
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))
import time
import psmove
import datetime
from csv import writer
from time import gmtime, strftime

if psmove.count_connected() < 1:
    print('No controller connected')
    sys.exit(1)

move = psmove.PSMove()

if move.connection_type == psmove.Conn_Bluetooth:
    print('bluetooth')
elif move.connection_type == psmove.Conn_USB:
    print('usb')
else:
    print('unknown')

if move.connection_type != psmove.Conn_Bluetooth:
    print('Please connect controller via Bluetooth')
    sys.exit(1)

# Called for every client connecting (after handshake)
def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    server.send_message_to_all("Hey all, a new client has joined us")

# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])


# Called when a client sends a message
def message_received(client, server, message):
    if len(message) > 200:
        message = message[:200]+'..'
    print("Client(%d) said: %s" % (client['id'], message))

gestures = ['tv-poke', 'shutter-right-left', 'letter-m', 'letter-y', 'fan-one-circle', 'fan-two-circles', 'no-gesture' ]
gesture_iterator = 0 # iterator for the list of gestures
samples_max = 10 # how many samples of each gesture to collect
sample_iterator = 0 # iterator within those samples
datapoints_max = 40 # how many data points to collect for each sample
datapoints_iterator = 0 # iterator within datapoints
changed = True

#print('server started')
#PORT = 9000
#server = WebsocketServer(PORT, host='localhost', loglevel=logging.INFO)
#server.set_fn_new_client(new_client)
#server.set_fn_client_left(client_left)
#server.set_fn_message_received(message_received)
#server.run_forever()

with open('motion-data-'+strftime("%Y-%m-%d-%H:%M:%S", gmtime())+'.csv', 'w', newline='') as f:
    data_writer = writer(f)
    data_writer.writerow(['gx', 'gy', 'gz', 'ax', 'ay', 'az', 'time', 'gesture_type'])
    while True:
        if (gesture_iterator == len(gestures)):
            print('all gestures gathered!')
            #server.send_message_to_all('{"gesture": "All gestures have been gathered!" }')

            sys.exit()
        
        gesture_type = str(gestures[gesture_iterator]) + '-' + str(sample_iterator)

        if (sample_iterator < 10):
            gesture_type = str(gestures[gesture_iterator]) + '-0' + str(sample_iterator)
        
        if (changed == True):
            print(gesture_type)

        changed = False
        # Get the latest input report from the controller
        while move.poll():
            pressed, released = move.get_button_events()
            if pressed & psmove.Btn_SQUARE:
                changed = True

        if changed:

            sample_iterator = sample_iterator + 1
            fill_empty_datapoints = datapoints_max - datapoints_iterator
            
            if (fill_empty_datapoints > 0):
                for x in range(0, fill_empty_datapoints):
                    data_writer.writerow([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, datetime.datetime.now(), gesture_type])

            if (sample_iterator >= samples_max):
                sample_iterator = 0
                gesture_iterator = gesture_iterator + 1
            if (gesture_iterator < len(gestures)):
                gesture_type = str(gestures[gesture_iterator]) + '-' + str(sample_iterator)
                if (sample_iterator < 10):
                    gesture_type = str(gestures[gesture_iterator]) + '-0' + str(sample_iterator)
            datapoints_iterator = 0

                #server.send_message_to_all('{"gesture": "%s" }' % gesture_type)

        trigger_value = move.get_trigger()
        #buttons = move.get_buttons()


        ax, ay, az = move.get_accelerometer_frame(psmove.Frame_SecondHalf)
        gx, gy, gz = move.get_gyroscope_frame(psmove.Frame_SecondHalf)

        if trigger_value > 20:
            if datapoints_iterator < datapoints_max:
                move.set_leds(0, 255, 0)
                move.update_leds()
                datapoints_iterator = datapoints_iterator+1
                data_writer.writerow([gx, gy, gz, ax, ay, az, datetime.datetime.now(), gesture_type])
            else :
                print('gesture duration exceeded')
                move.set_leds(255, 0, 0)
                move.update_leds()
            #server.send_message_to_all(
            #    '{ "gx": %6.2f, "gy": %6.2f, "gz": %6.2f, "ax": %5.2f, "ay": %5.2f, "az": %5.2f, "gesture": "%s" }' % (
            #        gx, gy, gz, ax, ay, az, gesture_type))
        if (trigger_value <= 20):
            move.set_leds(0, 0, 0)
            move.update_leds()
        #if buttons & psmove.Btn_SQUARE:
            #move.set_rumble(trigger_value)

        #else:
        #    move.set_rumble(0)



        #print('accel:', (move.ax, move.ay, move.az))
        #print('gyro:', (move.gx, move.gy, move.gz))
        #print('magnetometer:', (move.mx, move.my, move.mz))

        #print ('G: %6.2f %6.2f %6.2f' % (gx, gy, gz))

        #server.send_message_to_all('{"gx": '+ str(move.gx) + ', "gy": '+ str(move.gy) + ', "gz": '+ str(move.gz) + ', "ax": '+ str(move.ax)+  ', "ay": '+ str(move.ay)+ ', "az": '+ str(move.az)+'}')
        time.sleep(0.05)



