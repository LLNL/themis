def post_run():
    print('IN POST RUN')
    val = 1e99
    try:
        with open('output_deck', 'r') as fh:
            lines = fh.readlines()
            print(lines)
            for line in lines:
                toks = line.split()
                print(toks)
                if toks[0] == 'Y':
                    print(toks[0])
                    val = float(toks[2])
                    print(val)
    except Exception as e:
        print(e)
    finally:
        return val
