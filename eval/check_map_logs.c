/* The pinned vector Dict retains key pointers and has an initial capacity of 32. */
#include "binding.c"
#include <assert.h>
int main(void) {
    Log log={0};log.perf=0.75f;
    for (int map=0;map<BOPL_COUNTER_MAPS;map++) log.map_fraction[map]=(float)(map+1)/7750.0f;
    Dict *out=create_dict(32);my_log(&log,out);dict_set(out,"n",64);
    assert(out->size==142 && dict_get(out,"perf")->value==0.75);
    for (int map=0;map<BOPL_COUNTER_MAPS;map++) {
        char key[32];snprintf(key,sizeof(key),"map_%d_fraction",map);
        assert(dict_get(out,key)->value==log.map_fraction[map]);
    }
    free(out->items);free(out);
    puts("MAP_LOG_KEYS_AND_CAPACITY_OK");
}
