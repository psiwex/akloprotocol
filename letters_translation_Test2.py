def translate_morse(morse_array):
    translation = ""
    current_letter = ""
    i = 0

    dictionary = {
        ".-":"a", "-...":"b", "-.-.":"c", "-..":"d",
        ".":"e", "..-.":"f", "--.":"g", "....":"h",
        "..":"i", ".---":"j", "-.-":"k", ".-..":"l",
        "--":"m", "-.":"n", "---":"o", ".--.":"p",
        "--.-":"q", ".-.":"r", "...":"s", "-":"t",
        "..-":"u", "...-":"v", ".--":"w", "-..-":"x",
        "-.--":"y", "--..":"z", ".----": "1", "..---":"2",
        "...--":"3", "....-":"4", ".....":"5", "-....":"6",
        "--...":"7", "---..":"8", "----.":"9", "-----":"0"
    }

    while i < len(morse_array):

        if morse_array[i] == "||":  # Word boundary
            translation += dictionary.get(current_letter, "?")
            translation += " "
            current_letter = ""

        elif morse_array[i] == "|":  # Letter boundary
            translation += dictionary.get(current_letter, "?")
            current_letter = ""

        else:  # Dot or dash
            current_letter = current_letter + morse_array[i]

        i += 1

    # Add final letter if exists
    if current_letter:
        translation += dictionary.get(current_letter, "?")

    return translation

if __name__ == "__main__":
    
    # a
    test1 = [".", "-"] 
     # hi
    test2 = [".", ".", ".", ".", "|", ".", "."]
    # how are you
    test3 = [".", ".", ".", ".", "|", "-", "-", "-", "|", ".", "-", "-", "||", ".", "-", "|", ".", "-", ".", "|", ".", "||", 
             "-", ".", "-", "-", "|", "-", "-", "-", "|", ".", ".", "-"]  
    # 12345
    test4 = [".", "-", "-", "-", "-", "|", ".", ".", "-", "-", "-", "|", ".", ".", ".", "-", "-", "|", 
             ".", ".", ".", ".", "-", "|", ".", ".", ".", ".", "."] 
    # ?bc
    test5 = [".", "-", ".", "-", ".", "-", "|", "-", ".", ".", ".", "|", "-", ".", "-", "."] 

    print("Morse Code Translation for test 1:", translate_morse(test1), '\n')
    print("Morse Code Translation for test 2:", translate_morse(test2), '\n')
    print("Morse Code Translation for test 3:", translate_morse(test3), '\n')
    print("Morse Code Translation for test 4:", translate_morse(test4), '\n')
    print("Morse Code Translation for test 5:", translate_morse(test5), '\n')
