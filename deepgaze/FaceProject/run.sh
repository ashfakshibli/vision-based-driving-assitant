#!/bin/bash

commands () {
    /home/ashfak/.virtualenvs/cv/bin/python /home/ashfak/Desktop/DriversAssistanceSystem/deepgaze/FaceProject/plot_ex.py
    $SHELL # keep the terminal open after the previous commands are executed
}

export -f commands

gnome-terminal -e "bash -c 'commands'"
