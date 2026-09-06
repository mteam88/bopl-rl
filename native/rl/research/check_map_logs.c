/* The pinned vector Dict retains key pointers and has an initial capacity of 32. */
#include "../puffer/binding.c"
#include <assert.h>
int main(void) {
    Log log={0};log.perf=0.75f;
    for (int map=0;map<31;map++) log.map_fraction[map]=(float)(map+1)/496.0f;
    Dict *out=create_dict(32);my_log(&log,out);dict_set(out,"n",64);
    assert(out->size==39 && dict_get(out,"perf")->value==0.75);
    for (int map=0;map<31;map++) {
        char key[32];snprintf(key,sizeof(key),"map_%d_fraction",map);
        assert(dict_get(out,key)->value==log.map_fraction[map]);
    }
    free(out->items);free(out);
    puts("MAP_LOG_KEYS_AND_CAPACITY_OK");
}
