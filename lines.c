#include <stdio.h>
#include <stdlib.h>

#define MAX_CHARS 100

unsigned int count_lines(const char **filenames) {
    if (!filenames) return 0;

    unsigned int lines = 0, i = 0;
    char *line = (char*)malloc(MAX_CHARS * sizeof(char));
    if (!line)
        return 0;

    while (filenames[i]) {
        FILE *fp = fopen(filenames[i], "r");
        if (!fp) {
            i++;
            continue;
        }

        while (fgets(line, MAX_CHARS, fp))
            lines++;

        fclose(fp);
        i++;
    }

    free(line);
    return lines;
}

int main(void) {
    const char *filenames[] = { 
        "src/arithmetic.hpp",
        "src/bit.hpp", 
        "src/compare.hpp", 
        "src/data.hpp", 
        "src/linear.hpp", 
        "src/logical.hpp", 
        "src/operators.hpp", 
        "src/tensor.hpp", 
        "src/tensorbase.hpp", 
        "src/types.hpp",
        "src/constructors.hpp",
        NULL
    };

    unsigned int lines = count_lines(filenames);
    printf("Lines of code: %u\n", lines);
    return 0;
}