#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- Type Definitions --- */

// Define a function pointer type for our "method"
typedef void (*DisplayFunc)(const char* prefix, const char* target);

// Struct acting as a "Class" to handle the message logic
typedef struct {
    char* prefix;
    DisplayFunc execute;
} HelloWorldEmitter;

// Struct to manage an array of emitters
typedef struct {
    HelloWorldEmitter** workers;
    size_t count;
    size_t capacity;
} ExecutionManager;

/* --- Logic Implementation --- */

// The actual function that performs the printing
void standard_display(const char* prefix, const char* target) {
    if (prefix && target) {
        printf("%s, %s!\n", prefix, target);
    }
}

// "Constructor" for the Emitter
HelloWorldEmitter* create_emitter(const char* prefix) {
    HelloWorldEmitter* emitter = (HelloWorldEmitter*)malloc(sizeof(HelloWorldEmitter));
    if (!emitter) return NULL;

    emitter->prefix = strdup(prefix); // Dynamically allocate string copy
    emitter->execute = standard_display;
    return emitter;
}

// "Constructor" for the Manager
ExecutionManager* create_manager(size_t initial_cap) {
    ExecutionManager* mgr = (ExecutionManager*)malloc(sizeof(ExecutionManager));
    if (!mgr) return NULL;

    mgr->workers = (HelloWorldEmitter**)malloc(sizeof(HelloWorldEmitter*) * initial_cap);
    mgr->count = 0;
    mgr->capacity = initial_cap;
    return mgr;
}

// Add an emitter to the manager's list
void add_worker(ExecutionManager* mgr, HelloWorldEmitter* worker) {
    if (mgr->count < mgr->capacity) {
        mgr->workers[mgr->count++] = worker;
    }
}

// Cleanup function to prevent memory leaks
void destroy_manager(ExecutionManager* mgr) {
    for (size_t i = 0; i < mgr->count; i++) {
        free(mgr->workers[i]->prefix);
        free(mgr->workers[i]);
    }
    free(mgr->workers);
    free(mgr);
}

/* --- Main Execution --- */

int main(int argc, char* argv[]) {
    // Initialize the execution pipeline
    ExecutionManager* manager = create_manager(5);
    if (!manager) return EXIT_FAILURE;

    // Create and register the "Hello" worker
    HelloWorldEmitter* hello_worker = create_emitter("Hello");
    if (hello_worker) {
        add_worker(manager, hello_worker);
    }

    const char* payload = "World";

    // Iterate through workers using function pointers
    for (size_t i = 0; i < manager->count; i++) {
        HelloWorldEmitter* w = manager->workers[i];
        w->execute(w->prefix, payload);
    }

    // Explicit memory deallocation (Mandatory in C)
    destroy_manager(manager);

    return EXIT_SUCCESS;
}







