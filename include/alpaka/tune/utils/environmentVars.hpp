//
// Created by tim on 18.03.25.
//

#ifndef ENVIRONMENTVARS_H
#define ENVIRONMENTVARS_H
static std::size_t getReRuns()
{
    if(char const* var = std::getenv("TunerReRuns"))
    {
        try
        {
            std::size_t const value = static_cast<std::size_t>(std::stoul(var));
            return value;
        }
        catch(std::exception const& e)
        {
            std::cerr << "Invalid value for TunerReRuns: " << e.what() << std::endl;
        }
    }
    return 0;
}
#endif // ENVIRONMENTVARS_H
