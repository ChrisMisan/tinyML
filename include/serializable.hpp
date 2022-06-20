#pragma once

#include <common.hpp>

namespace tinyML
{
    class serializable
    {
    public:
        virtual ~serializable() = default;
    public:
        [[nodiscard]] virtual nlohmann::json serialize() const = 0;
        virtual void deserialize(const nlohmann::json& data) = 0;
    };
}