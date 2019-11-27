# rnn


<b>Dataset(lotr.txt)</b><br>
</n>Text files of The Hobbit, The Lord of the Rings: Fellowship of the Ring, The Lord of the Rings: The Two Towers, and the Lord of the Rings: The Return of the King that were compiled into one large text file. I took the text from eBooks that I bought and ran through a program that extracted the text from the eBooks into text files.
  
<b>Model(lotr.py)</b><br>
A Recurrent Neural Network built with an Embedding Layer for input, three Dropout layers, and two LSTM layers between the Dropout layers.
This was built with the Tensorflow API and Python. I also used the Numpy, os, and time libraries.

<b>Output(output.txt)</b><br>
1000 character output file using a start string "Gandalf" and the compiled model. 
