### ESP32-cam：

#### cam/main_process.py：

function：get picture&recognition 

receive a signal from PC(key 3) and start,result send to PC

### STC-B：

display "WELCOME"  lux<20:stop working,send"**0x CC..**",display“ERROR”

1，press key 1:"IN" press key 2：“OUT”  display&voice,

2，if press key 3:①BEEP&display"WORKING" ②send **0x AA 55 00** to PC(start get picture)

receive **0x BB..**:BEEP，dislpay city,id（nixie tube）

### PC：

#### from stcb(command):

receive  **0x AA 55 00**: simulate "press Enter" and cam start get pic

#### from cam(result):

receive recognition results(XX-XXXX-XXX-XXX&object):①store in database ②send **0x BB...** to stcb

③show on vis system





##### **Serial **Definition：

###### 1,**0xAA:STCB->PC**

**0xAA,0x55 start getting pic **

**0xAA,0x44：in **

**0xAA,0x33：out**

###### 2,**0xBB：PC->STCB result**

###### 3,**0xCC :warning:error**

