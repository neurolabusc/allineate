#include <math.h>
#include <stdio.h>
#include <string.h>

#include "coreg_fast_nms.h"

static void set_candidate(cf_nms_candidate *c, double cost, double rx_deg,
                          size_t order) {
    memset(c, 0, sizeof(*c));
    c->c = cost;
    c->p[3] = rx_deg;
    c->order = order;
}

static int separated(const cf_nms_candidate *c, int n, double floor_deg) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double d0 = c[i].p[3] - c[j].p[3];
            double d1 = c[i].p[4] - c[j].p[4];
            double d2 = c[i].p[5] - c[j].p[5];
            double d = sqrt(d0*d0 + d1*d1 + d2*d2);
            if (d < floor_deg) return 0;
        }
    return 1;
}

int main(void) {
    cf_nms_candidate pool[7], out[6];

    /* The historical top-N helper remains strictly cost sorted. Equal costs do
       not displace earlier enumeration, preserving the original grid order. */
    memset(out, 0, sizeof(out));
    for (int i = 0; i < 3; i++) out[i].c = 1.0e9;
    double p[12] = {0};
    p[3] = 30.0; cf_insert_sorted_candidate(out, 3, p, 2.0, 0);
    p[3] = 10.0; cf_insert_sorted_candidate(out, 3, p, 1.0, 1);
    p[3] = 20.0; cf_insert_sorted_candidate(out, 3, p, 1.0, 2);
    if (out[0].order != 1 || out[1].order != 2 || out[2].order != 0) {
        fprintf(stderr, "top-N insertion failed cost/tie ordering regression\n");
        return 1;
    }

    /* A late, better point overlaps TWO previously plausible basins. In-place
       replacement would leave the 15-degree neighbour and violate the 20-degree
       separation. Sorted greedy NMS must retain only the better basin. */
    set_candidate(&pool[0], 1.0,  0.0, 0);
    set_candidate(&pool[1], 2.0, 30.0, 1);
    set_candidate(&pool[2], 3.0, 60.0, 2);
    set_candidate(&pool[3], 0.5, 15.0, 3);
    set_candidate(&pool[4], 4.0, 90.0, 4);
    set_candidate(&pool[5], 5.0, 120.0, 5);
    set_candidate(&pool[6], 6.0, 150.0, 6);
    int n = cf_select_angular_nms(pool, 7, out, 6, 20.0, 1.0, 1.0e9);
    if (n != 5 || out[0].c != 0.5 || out[0].p[3] != 15.0 ||
        !separated(out, n, 20.0)) {
        fprintf(stderr, "NMS failed ordering/separation regression\n");
        return 1;
    }

    /* Equal costs retain enumeration order, independent of qsort internals. */
    set_candidate(&pool[0], 1.0, 80.0, 2);
    set_candidate(&pool[1], 1.0, 40.0, 0);
    set_candidate(&pool[2], 1.0,  0.0, 1);
    n = cf_select_angular_nms(pool, 3, out, 3, 20.0, 1.0, 1.0e9);
    if (n != 3 || out[0].order != 0 || out[1].order != 1 || out[2].order != 2) {
        fprintf(stderr, "NMS failed deterministic tie-order regression\n");
        return 1;
    }

    /* Separation is three-dimensional, not just the first rotation axis. */
    set_candidate(&pool[0], 0.1, 0.0, 0);
    set_candidate(&pool[1], 0.2, 0.0, 1); pool[1].p[4] = 10.0;
    set_candidate(&pool[2], 0.3, 0.0, 2); pool[2].p[5] = 25.0;
    n = cf_select_angular_nms(pool, 3, out, 3, 20.0, 1.0, 1.0e9);
    if (n != 2 || out[0].order != 0 || out[1].order != 2 ||
        !separated(out, n, 20.0)) {
        fprintf(stderr, "NMS failed multi-axis separation regression\n");
        return 1;
    }

    /* Invalid thresholds/costs fail closed and never duplicate the best point. */
    if (cf_select_angular_nms(pool, 3, out, 3, 0.0, 1.0, 1.0e9) != 0) {
        fprintf(stderr, "NMS accepted zero angular separation\n");
        return 1;
    }
    set_candidate(&pool[0], NAN, 0.0, 0);
    set_candidate(&pool[1], 1.0e9, 30.0, 1);
    if (cf_select_angular_nms(pool, 2, out, 2, 20.0, 1.0, 1.0e9) != 0) {
        fprintf(stderr, "NMS accepted non-finite/penalty costs\n");
        return 1;
    }

    puts("coreg_fast NMS tests passed");
    return 0;
}
