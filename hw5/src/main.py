import DP
import CTF
import DF
import test_RMLR
import sys
def main():
    # read the train file from first arugment
    option = sys.argv[1]

    #Data Preprocessing
    DP.train()
    DP.dev()
    DP.test()

    if option == '0':
        # Make top-2000 CTF vectors
        CTF.train()
        CTF.dev()
        CTF.test()

        #train result
        test_RMLR.CTF_final()
    elif option == '1':
        # Make top-2000 DF vectors
        DF.train()
        DF.dev()
        DF.test()

        # train result
        test_RMLR.DF_final()

# Main entry point to the program
if __name__ == '__main__':
    main()