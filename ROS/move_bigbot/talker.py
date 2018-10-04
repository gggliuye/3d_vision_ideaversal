#!/usr/bin/env python

import rospy
import sys
from AlphaBot2 import AlphaBot2
from ideaversal.msg import move

def talker():
    pub = rospy.Publisher('commander', move, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    count = 0
    while not rospy.is_shutdown() and count<10000:
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
        count = count + 1

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass