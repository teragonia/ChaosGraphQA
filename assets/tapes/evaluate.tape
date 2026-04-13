Output assets/evaluate.gif

Set Shell "bash"
Set FontSize 13
Set Width 1200
Set Height 600
Set Theme "Catppuccin Mocha"
Set Padding 16
Set TypingSpeed 40ms

Hide
Type "export ANTHROPIC_API_KEY=$(cat /tmp/cgqa_api_key)"
Enter
Type "export PS1='$ '"
Enter
Type "cgqa generate --generator-type multihop --complexity 2 --num-questions 2 --seed 42 --output /tmp/demo.json"
Enter
Sleep 5s
Hide
Type "clear"
Enter
Show

Sleep 500ms

# Assumes demo.json already generated (run generate.tape first)
Type "cgqa list-models"
Sleep 500ms
Enter
Sleep 5s

Type "cgqa evaluate /tmp/demo.json --model anthropic/claude-haiku-4.5 --output /tmp/results.json"
Sleep 500ms
Enter
Sleep 12s

Type "cgqa analyze /tmp/results.json"
Sleep 500ms
Enter
Sleep 6s
