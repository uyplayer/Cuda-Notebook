
#include <iostream>
#include "cjson/cJSON.h"
#include <cstdlib>
#include <cstdio>

int main(){
    char versionBuf[32];
    snprintf(versionBuf, sizeof(versionBuf), "%d.%d.%d", CJSON_VERSION_MAJOR, CJSON_VERSION_MINOR, CJSON_VERSION_PATCH);
    cJSON *versionJson = cJSON_CreateString(versionBuf);
    cJSON *rootJson = cJSON_CreateObject();
    cJSON_AddItemToObject(rootJson, "version", versionJson);
    char *jsonStr = cJSON_Print(rootJson);
    printf("%s\n", jsonStr);
    free(jsonStr);
    cJSON_Delete(rootJson);


    std::cout << "conan package managers passed" << std::endl;

    return 0;
}