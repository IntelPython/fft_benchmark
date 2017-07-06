
    {
        const char* s = getenv("N1");
        N1 = (s) ? atoi(s) : 0;
        if (N1 < 1) N1 = 2048;

        s = getenv("N2");
        N2 = (s) ? atoi(s) : 0;
        if (N2 < 1) N2 = 4096;

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
    printf("TAG: Size: (%d, %d), Reps: %d, Samples: %d\n", N1, N2, reps, samps);
    printf("TAG: ");
