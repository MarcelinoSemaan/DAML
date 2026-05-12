#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>

// Utilizing a custom namespace to encapsulate core logic
namespace Core Systems {

    // A template-based base class to handle message processing
    template <typename T>
    class MessageProcessor {
    public:
        virtual ~MessageProcessor() = default;
        virtual void process(const T& data) const = 0;
    };

    // A concrete implementation for String-based communication
    class HelloWorldEmitter : public MessageProcessor<std::string> {
    private:
        std::string prefix;

    public:
        explicit HelloWorldEmitter(std::string  p) : prefix(std::move(p)) {}

        void process(const std::string& data) const override {
            std::cout << prefix << " " << data << "!" << std::endl;
        }
    };

    // Manager class to handle the lifecycle of the operation
    class ExecutionManager {
    private:
        std::vector<std::unique_ptr<MessageProcessor<std::string>>> workers;

    public:
        void addWorker(std::unique_ptr<MessageProcessor<std::string>> worker) {
            workers.push_back(std::move(worker));
        }

        void run(const std::string& target) {
            // Using a lambda with std::for_each for modern iteration
            std::for_each(workers.begin(), workers.end(), [&target](const auto& worker) {
                if (worker) {
                    worker->process(target);
                }
            });
        }
    };
}

/**
 * Main Entry Point
 * Complexity: O(N) where N is the number of workers registered
 */
int main() {
    using namespace CoreSystems;

    // Initializing the manager
    auto manager = std::make_unique<ExecutionManager>();

    // Injecting dependencies
    manager->addWorker(std::make_unique<HelloWorldEmitter>("Hello"));
    
    // Abstracting the payload
    const std::string payload = "World";

    try {
        // Execute the processing pipeline
        manager->run(payload);
    } catch (const std::exception& e) {
        std::cerr << "Execution Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}