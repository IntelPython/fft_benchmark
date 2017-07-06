
    {
        const char* s = getenv("N");
        N = (s) ? atoi(s) : 0;
        if (N < 1) N = 1000000;

        s = getenv("REPS");
        reps = (s) ? atoi(s) : 0;
        if (reps < 1) reps = 1;

        s = getenv("S");
        samps = (s) ? atoi(s) : 0;
        if (samps < 1) samps = 1;
    }

    {
        char version[DFTI_VERSION_LENGTH];

        DftiGetValue(hand, DFTI_VERSION, version);
        printf("TAG: %s\n", version);
    }
#ifdef DESCRIPTION_STR
    printf("TAG: "DESCRIPTION_STR"\n");
#endif
    printf("TAG: Size: %d, Reps: %d, Samples: %d\n", N, reps, samps);
    printf("TAG: ");
