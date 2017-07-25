sleep 2
a=0
say "Show me something"
while [ $a -lt 2 ]
do
	
	python exp.py > example.txt
	if grep -Fxq "1" example.txt
	then 
		say "You are shaking too much. Please show it again"
	else
		say "I see"
		
		say -f example.txt
		clear
		cat example.txt
		a=10
		
	fi
done
