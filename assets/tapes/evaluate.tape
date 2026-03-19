Output ../evaluate.gif

Set Shell "bash"
Set FontSize 13
Set Width 110
Set Height 36
Set Theme "Catppuccin Mocha"
Set Padding 16
Set TypingSpeed 40ms

Hide
Type "export PS1='$ '"
Enter
Show

Sleep 500ms

# Assumes demo.json already generated (run generate.tape first)
Type "cgqa list-models"
Sleep 500ms
Enter
Sleep 2s

Type "cgqa evaluate /tmp/demo.json --model anthropic/claude-3.7-sonnet --output /tmp/results.json"
Sleep 500ms
Enter
Sleep 8s

Type "cgqa analyze /tmp/results.json"
Sleep 500ms
Enter
Sleep 3s
