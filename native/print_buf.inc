    if (buf && N < 11) {
        for(i = 0; i < N; i++) {
            printf("(%.8g, %.8g)\n", buf[i].real, buf[i].imag);
        }
    }
