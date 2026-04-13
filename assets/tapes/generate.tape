Output assets/generate.gif

Set Shell "bash"
Set FontSize 13
Set Width 1200
Set Height 550
Set Theme "Catppuccin Mocha"
Set Padding 16
Set TypingSpeed 40ms

# Prompt
Hide
Type "export PS1='$ '"
Enter
Show

Sleep 500ms

Type "cgqa generate --generator-type multihop --complexity 2 --num-questions 5 --seed 42 --output /tmp/demo.json"
Sleep 500ms
Enter
Sleep 4s

Type "cgqa generate --generator-type hierarchical --complexity 3 --num-questions 5 --seed 7"
Sleep 500ms
Enter
Sleep 4s

Type "cgqa info /tmp/demo.json"
Sleep 500ms
Enter
Sleep 3s
