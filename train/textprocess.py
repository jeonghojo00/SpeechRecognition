import torch

class TextTransform:
    """ Maps Characters -> Integers and vice versa"""
    def __init__(self):
        char_map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char2index = {}
        self.index2char = {}
        for line in char_map_str.strip().split('\n'):
            char, index = line.split()
            self.char2index[char] = int(index)
            self.index2char[int(index)] = char
        self.index2char[1] = ' '

    def text_to_int(self, text):
        """ Convert text into an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char2index['<SPACE>']
            else:
                ch = self.char2index[c]
            int_sequence.append(ch)
        return int_sequence
    
    def int_to_text(self, labels):
        """ Convert integer labels to a text sequenc """
        text = []
        for i in labels:
            text.append(self.index2char[i])
        return ''.join(text).replace('<SPACE>', ' ')
    
text_transform = TextTransform()

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):
    """Decodes a batch of output into a batch of texts"""
    #output :  (batch, time, n_class)
    #labels :  (batch, time)
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text_sequence(
				labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text_sequence(decode))
        return decodes, targets

def Decoder(output, blank_label=28, collapse_repeated=True):
    """Decodes single output into a text for inference"""    
    arg_maxes = torch.argmax(output, dim=2)
    for i, args in enumerate(arg_maxes):
        decode = []
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decoded = text_transform.int_to_text(decode)
    return decoded