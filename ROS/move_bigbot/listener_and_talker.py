#!/usr/bin/env python

import rospy
import sys
from AlphaBot2 import AlphaBot2
from ideaversal.msg import move
import time

bot = AlphaBot2()

def callback(mov):
    if mov.command == 8:
        bot.forward()
    elif mov.command == 4:
        bot.left()
    elif mov.command == 6:
        bot.right()
    elif mov.command == 2:
        bot.backward()
    elif mov.command == 5:
        bot.stop()
    else:
        print('[ERROR] command error !')
    time.sleep(mov.time)
    # bot.stop()

def listener():
    rospy.init_node('mover', anonymous=True)
    pub = rospy.Publisher('commander', move, queue_size=1)
    sub = rospy.Subscriber('commander', move, callback)
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        var = input("Please enter direction: ")
        # var = var.split(' ')
        com = int(var)
        time = 1
        command_str = 'move the robot '+str(com)+' for '+str(time)
        rospy.loginfo(command_str)
        msg = move()
        msg.command = com
        msg.time = time
        pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    print(' ---- start ---- ')
    listener()