/*
    Copyright 2005-2013 Intel Corporation.  All Rights Reserved.

    The source code contained or described herein and all documents related
    to the source code ("Material") are owned by Intel Corporation or its
    suppliers or licensors.  Title to the Material remains with Intel
    Corporation or its suppliers and licensors.  The Material is protected
    by worldwide copyright laws and treaty provisions.  No part of the
    Material may be used, copied, reproduced, modified, published, uploaded,
    posted, transmitted, distributed, or disclosed in any way without
    Intel's prior express written permission.

    No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise.  Any license under such
    intellectual property rights must be express and approved by Intel in
    writing.
*/

#ifndef __TBB_tick_count_H
#define __TBB_tick_count_H
#include <assert.h>
#if _WIN32||_WIN64
#include <windows.h>
#elif __linux__
#include <time.h>
#else /* generic Unix */
#include <sys/time.h>
#endif /* (choice of OS) */

typedef long long moment_t;

inline moment_t moment_now( void )
{
    moment_t result;
#if _WIN32||_WIN64
    LARGE_INTEGER qpcnt;
    QueryPerformanceCounter(&qpcnt);
    result = qpcnt.QuadPart;
#elif __linux__
    struct timespec ts;
    int status = clock_gettime( CLOCK_REALTIME, &ts );
    assert( status == 0 );
    if( status != 0 ) abort();
    result = (long long)1000000000UL*(long long)ts.tv_sec + (long long)ts.tv_nsec;
#else /* generic Unix */
    struct timeval tv;
    int status = gettimeofday(&tv, NULL);
    assert( status == 0 );
    if( status != 0 ) abort();
    result = (long long)1000000*(long long)tv.tv_sec + (long long)tv.tv_usec;
#endif /*(choice of OS) */
    return result;
}

inline moment_t moment_from_seconds( double sec )
{
#if _WIN32||_WIN64
    LARGE_INTEGER qpfreq;
    QueryPerformanceFrequency(&qpfreq);
    return (long long)sec*qpfreq.QuadPart;
#elif __linux__
    return (long long)sec*1E9;
#else /* generic Unix */
    return (long long)sec*1E6;
#endif /* (choice of OS) */
}

inline double seconds_from_moment( moment_t value )
{
#if _WIN32||_WIN64
    LARGE_INTEGER qpfreq;
    QueryPerformanceFrequency(&qpfreq);
    return value/(double)qpfreq.QuadPart;
#elif __linux__
    return value*1E-9;
#else /* generic Unix */
    return value*1E-6;
#endif /* (choice of OS) */
}


#endif /* __TBB_tick_count_H */

