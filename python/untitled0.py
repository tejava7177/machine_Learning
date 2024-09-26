# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:17:23 2024

@author: user
"""

from datetime import datetime
import random

place=input("Where? ")
time=input("Time? ")
opponent=input("Team? ")
goals=input("Son's Goal? ")
aids=input("Assist? ")
score_me=input("Son's Team Goal? ")
score_you=input("opposing Team's Goal? ")


news="[프리미어 리그 속보("+str(datetime.now())+")]\n"

"""
news=news+"손흥민 선수는 "+place+"에서" +time+"에 열린 경기에 출전 "
news=news+"상대 팀은 "+opponent+"입니다"
"""

wording=["상대 팀을 이겼습니다.", "상대를 대상으로 통쾌한 승리를 거머쥐었습니다.", "상대 팀을 꺾었습니다. "]
news=news+"손흥민 선수 팀이 " + score_me+"골을 넣어" + score_you+"골을 넣은" + random.choice(wording)

if score_me>score_you:
    news=news+"손흥민 선수의 팀이" + score_me + "골을 넣어"+score_you +"골 넣은 상대팀을 이겼습니다."
elif score_me<score_you:
    news=news+"손흥민 선수의 팀이" + score_me+ "골을 넣어"+score_you + "골 넣은 상대팀에게 졌습니다."
else:
    news=news+"두 팀은 "+score_me+"대 " + score_you+"로 비겼습니다."
    
    


if(int(goals)>0 and int(aids)>0):
    news=news+"손흥민 선수는 " + goals+"골에 도움" +aids+"개로 승리를 이끌었습니다."
elif int(goals)>0 and int(aids)==0:
    news=news+"손흥민 선수는 " + goals+"개로 승리를 이끌었습니다."

elif int(goals)==0 and int(aids)>0:
    news=news+"손흥민 선수는 " + aids+"개로 승리를 이끌었습니다."
else:
    news=news+"아쉽게도 이번 경기에서 손흥민을 활약하지 못했습니다."


print(news)


from gtts import gTTS
import playsound

tts=gTTS(text=news, lang="ko")
tts.save("news_Son.mp3")
playsound.playsound("news_Son.mp3", True)


