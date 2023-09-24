#!/bin/bash

student=( 
	"h3538087"
	"fanasyev"
	"ggchan"
	"yhchang"
	"ycheung3"
	"ccheung4"
	"nyching"
	"kdjeow"
	"jmkim"
	"elau"
	"wylau2"
	"yclau"
	"kkleung"
	"ysli3"
	"knliu"
	"mlylok"
	"klu"
	"cslui"
	"zysu"
	"cswong"
	"mswong"
	"htyeung"
	"wyvying"
	"naliyeva"
	"cwchin"
	"cyhung"
	"hklam"
	"wylam"
	"mhluk"
	"ayush"
	"psharma"
	"zhtai"
	"yywu"
	"h3527459"
	"smdabaxi"
	"staetter"
	"thchan"
	"kcchang"
	"khchang"
	"bschawla"
	"cychiu"
	"hfmchu"
	"wlding"
)
student2=(
	"hdong"
	"vdua"
	"mfossier"
	"agupta"
	"pgupta"
	"andelwal"
	"bkwong"
	"cwlai"
	"xlai"
	"h3552485"
	"mtcman"
	"stang"
	"thcpoon"
	"rrrajpal"
	"esattar"
	"grshen"
	"irimanne"
	"myting"
	"shtsoi"
	"llnoefer"
	"tkwang"
	"scwong"
	"ykwong"
	"byang"
	"h3527394"
	"h3513467"
	"h3539323"
	"cnchan"
	"yhgao"
	"h3532746"
	"h3536724"
	"h3510017"
	"h3537222"
	"h3518214"
	"h3542533"
	"h3523257"
	"h3517898"
	"h3523401"
	"h3527737"
	"h3518666"
	"h3518673"
	"h3523465"
	"h3514196"

)

# for(( ip=0; ip < ${#student2[@]}; ip++ ))
# 	do
# 		wget https://i.cs.hku.hk/~${student2[$ip]}/lab7.zip -O ~/Downloads/0/${student2[$ip]}.zip
# 		# upzip ~/Downloads/0/${student2[$ip]}.zip
# 		# echo $ip
# 	done

for(( ip=0; ip < ${#student2[@]}; ip++ ))
do
	wget --spider -q -o /dev/null --tries=1 -T 5 https://i.cs.hku.hk/~${student2[$ip]}/lab7.zip  #<==接收函数的传参，即把结尾的$*传到这里。
	# upzip ~/Downloads/0/${student2[$ip]}.zip
	# echo $ip
	if [ $? -eq 0 ]
    then
        echo "!!! https://i.cs.hku.hk/~${student2[$ip]}/lab7.zip is yes.!!!"
    else
        echo "https://i.cs.hku.hk/~${student2[$ip]}/lab7.zip is fail."
    fi

    wget --spider -q -o /dev/null --tries=1 -T 5 https://i.cs.hku.hk/~${student2[$ip]}/lab7/lab7.zip  #<==接收函数的传参，即把结尾的$*传到这里。
	# upzip ~/Downloads/0/${student2[$ip]}.zip
	# echo $ip
	if [ $? -eq 0 ]
    then
        echo "!!! https://i.cs.hku.hk/~${student2[$ip]}/lab7/lab7.zip is yes.!!!"
    else
        echo "https://i.cs.hku.hk/~${student2[$ip]}/lab7/lab7.zip is fail."
    fi

    wget --spider -q -o /dev/null --tries=1 -T 5 https://i.cs.hku.hk/~${student2[$ip]}/Lab7/lab7.zip  #<==接收函数的传参，即把结尾的$*传到这里。
	# upzip ~/Downloads/0/${student2[$ip]}.zip
	# echo $ip
	if [ $? -eq 0 ]
    then
        echo "!!! https://i.cs.hku.hk/~${student2[$ip]}/Lab7/lab7.zip is yes.!!!"
    else
        echo "https://i.cs.hku.hk/~${student2[$ip]}/Lab7/lab7.zip is fail."
    fi

done




