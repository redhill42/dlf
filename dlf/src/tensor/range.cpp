#include <vector>
#include <tensor.h>

namespace dlf {
namespace {

class Parser {
    enum Token {
        NUM, COLON, COMMA, ELLIPSES, EOI, ERROR
    };

    const char* input;
    int num;
    Token token;
    std::vector<SliceDim> range;

    static void parse_error() {
        throw std::runtime_error("invalid slice range specification");
    }

    Token advance() {
        bool negate = false;

        while (*input == ' ')
            ++input;

        switch (*input) {
        case '\0':
            return EOI;
        case ':':
            ++input;
            return COLON;
        case ',':
            ++input;
            return COMMA;

        case '.':
            if (*++input == '.' && *++input == '.') {
                ++input;
                return ELLIPSES;
            }
            return ERROR;

        case '-':
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            if (*input == '-') {
                ++input;
                negate = true;
            }
            num = 0;
            while (*input >= '0' && *input <= '9') {
                num = num*10 + (*input - '0');
                ++input;
            }
            if (negate) num = -num;
            return NUM;

        default:
            return ERROR;
        }
    }

    Token next() {
        token = advance();
        return token;
    }

    SliceDim parse_dim() {
        int  start     = 0;
        int  end       = std::numeric_limits<int>::max();
        int  step      = 1;
        bool has_start = false;
        bool has_end   = false;

        if (token == NUM) {
            start = num;
            has_start = true;
            if (next() != COLON) {
                end = start + 1;
            }
        }
        if (token == COLON && next() == NUM) {
            end = num;
            has_end = true;
            next();
        }
        if (token == COLON && next() == NUM) {
            step = num;
            next();
        }
        if (step < 0 && !(has_start && has_end)) {
            start = std::numeric_limits<int>::max();
            end = std::numeric_limits<int>::lowest();
        }
        return SliceDim(start, end, step);
    }

public:
    Parser(const char* input) : input(input) {
        next();
    }

    std::vector<SliceDim> parse(size_t rank) {
        int fill_ind = -1;
        for (int idim = 0; idim < rank; idim++) {
            if (token == ELLIPSES) {
                next();
                if (fill_ind != -1)
                    parse_error();
                fill_ind = idim;
            } else {
                range.push_back(parse_dim());
            }
            if (token != COMMA)
                break;
            next();
        }

        if (token != EOI)
            parse_error();
        if (fill_ind != -1) {
            while (range.size() < rank) {
                range.insert(range.begin()+fill_ind, SliceDim{});
            }
        }
        return range;
    }
};

} // anonymous namespace

std::vector<SliceDim> parse_slice_range(const char* spec, size_t rank) {
    Parser parser(spec);
    return parser.parse(rank);
}

} // namespace dlf
