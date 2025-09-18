#include "STC15F2K60S2.H"        //必须。
#include "sys.H"                 //必须。
#include "Uart1.H" 
#include "displayer.H" 
#include "key.H" 
#include "beep.H" 
#include "adc.h"
code unsigned long SysClock=11059200;         //必须。定义系统工作时钟频率(Hz)，用户必须修改成与实际工作频率（下载时选择的）一致
#ifdef _displayer_H_                          //显示模块选用时必须。（数码管显示译码表，用戶可修改、增加等） 
code char decode_table[]={
	       0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f,0x00,0x08,0x40,0x01,0x41,0x48,0x76,0x00,0x00,0x00,
 /* ??:   0  1    2	   3    4	   5    6	   7    8	   9	  10	 11		12   13   14   15   16   17   18   19  */
 /* ??:   0  1    2    3    4    5    6    7    8    9   (?)  ?-  ?-  ?-  ??-  ??-                 */  
	       0x3f|0x80,0x06|0x80,0x5b|0x80,0x4f|0x80,0x66|0x80,0x6d|0x80,0x7d|0x80,0x07|0x80,0x7f|0x80,0x6f|0x80,  
 /* ??:   20        21        22        23        24        25        26         27        28        29     */
 /* ???? 0         1         2         3         4         5         6          7         8         9      */
			   0x77,0x7c,0x39,0x5E,0x79,0x71,0x3D,0x76,0x30,0x1E,0x75,0x38,0x15,0x37,0x3F,0x73,0x6B,0x33,0x6D,0x31,0x3E,0x3E,0x2A,0x76,0x66,0x5B,
 /* ??:30   31   32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47   48   49   50   51   52   53   54   55   */
 /*??		A		 B    C    D    E    F    G    H    I    J    K    L    M    N    O    P    Q    R    S    T    U    V    W    X    Y    Z    */
				 0x37|0x80,0x79|0x80,0x5E|0x80,0x6D|0x80,0x30|0x80,0x31|0x80,0x37|0x80
 /* ??:    56        57        58        59        60        61        62    */  
 /*??      N.        E.        d.        s.        I.        t.        n.    */
 /*???? Mon.      TuE.      Wed.     Thurs.     FrI.      Sat.      Sun.   */
};
#endif

unsigned char start_cam[5]={0xAA,0x55,0x00};//按K3，开始识别
unsigned char in[5]={0xAA,0x44,0x00};//开始入库
unsigned char out[5]={0xAA,0x33,0x00};//开始入库
unsigned char luxstop[5]={0xCC,0x55,0x01};//光照低，停止识别
unsigned char luxstart[5]={0xCC,0x55,0x02};//开始识别的信息
unsigned char recstore[5];//接收到的存在这
unsigned char recmask[1]={0xBB};//接收到成功的包头BB开始
char a=10;
char state=1;//控制显示，0：error 1：welcome 2：城市月日
unsigned char timer_5s = 0;//5s计时器
unsigned char display_data[8];//识别成功后显示的
void my1S_callback(){
	if (++timer_5s >= 5) {
		timer_5s = 0;
		state=1;
		if(GetADC().Rop<20){
			SetBeep(3000,150);
			state=0;
			Uart1Print(luxstop,sizeof(luxstop));
		}
		
	}
}
void myNav_callback()
{ char k;
	k=GetAdcNavAct(enumAdcNavKey3);
	if( k == enumKeyPress ) {
		SetBeep(1800,20);
		
		Uart1Print(start_cam,sizeof(start_cam));
}}
void mykey_callback()
{
	char r;
	char g;
	r=GetKeyAct(enumKey1);
	g=GetKeyAct(enumKey2);
	if( r == enumKeyPress ) {
		SetBeep(1800,20);
		state=3;
		Uart1Print(in,sizeof(in));
}
	if( g == enumKeyPress ) {
		SetBeep(1800,20);
		state=4;
		Uart1Print(out,sizeof(out));
}
}
void my10mS_callback()

{ 
	if (state==0)
	Seg7Print(a,34,47,47,44,47,a,a);
	if (state==1)
	Seg7Print(52,34,41,32,44,42,34,a);
	if (state==2)
	Seg7Print(display_data[0], display_data[1], display_data[2], display_data[3],
			  display_data[4], display_data[5], display_data[6], display_data[7]);
	if (state==3)
		Seg7Print(43,44,52,a,a,38,43,a);
	if (state==4)
		Seg7Print(43,44,52,a,a,44,50,49);
}
void rec_callback()
{
	char city;
	char month;
	char day;
	// 8位数码管
	

	city = recstore[1];   
	month = recstore[2];  
	day = recstore[3];   
	state=2;
	SetBeep(4000,5);
	switch(city)
	{
		case 0x01: // ChangSha 
			display_data[0] = 32; // 'C'
			display_data[1] = 48; // 'S'
			break;
		case 0x02: // XiangTan 
			display_data[0] = 53; // 'X'
			display_data[1] = 49; // 'T'
			break;
		case 0x03: // ZhuZhou 
			display_data[0] = 55; // 'Z'
			display_data[1] = 55; // 'Z'
			break;
		case 0x04: // HengYang 
			display_data[0] = 37; // 'H'
			display_data[1] = 54; // 'Y'
			break;
		case 0x05: // ShaoYang 
			display_data[0] = 48; // 'S'
			display_data[1] = 54; // 'Y'
			break;
		case 0x06: // YueYang 
			display_data[0] = 54; // 'Y'
			display_data[1] = 54; // 'Y'
			break;
		case 0x07: // ZhangJiaJie
			display_data[0] = 55; // 'Z'
			display_data[1] = 39; // 'J'
			break;
		default: // "Err"
			display_data[0] = 34; // 'E'
			display_data[1] = 47; // 'r'
			display_data[2] = 47; // 'r'
			break;
	}


	display_data[2] = 12; // 分隔符


	display_data[3] = month / 10; 
	display_data[4] = month % 10; 
	

	display_data[5] = 12; // '-'
	

	display_data[6] = day / 10;   
	display_data[7] = day % 10;   
	
	
}
void main() 
{ 
  Uart1Init(9600);
	DisplayerInit();  
	KeyInit();
	BeepInit();
	AdcInit();
	SetDisplayerArea(0,7);	
  //SetUart1Rxd(rxdbuf, sizeof(rxdbuf), 0, 0);	
	//SetEventCallBack(enumEventUart1Rxd, myrxd1);
	SetUart1Rxd(recstore, 5, recmask, 1);	
	SetEventCallBack(enumEventUart1Rxd,rec_callback);
	SetEventCallBack(enumEventKey,mykey_callback);
	SetEventCallBack(enumEventNav, myNav_callback);
	SetEventCallBack(enumEventSys1S, my1S_callback);
	SetEventCallBack(enumEventSys10mS, my10mS_callback);
  MySTC_Init();	 
	while(1)             	
		{ MySTC_OS();    
		}	             
}