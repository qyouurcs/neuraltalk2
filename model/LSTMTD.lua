local LSTMTD= {}

local ok, cunn = pcall(require, 'cunn')
LookupTable = nn.LookupTable

function LSTMTD.lstmtdnn(input_encoding_size, word_vocab_size, word_vec_size, char_vocab_size, char_vec_size,
                      feature_maps, kernels, length, use_words, use_chars, batch_norm, highway_layers)
    -- n = number of layers
    -- word_vocab_size = num words in the vocab    
    -- word_vec_size = dimensionality of word embeddings
    -- char_vocab_size = num chars in the character vocab
    -- char_vec_size = dimensionality of char embeddings
    -- feature_maps = table of feature map sizes for each kernel width
    -- kernels = table of kernel widths
    -- length = max length of a word
    -- use_words = 1 if use word embeddings, otherwise not
    -- use_chars = 1 if use char embeddings, otherwise not
    -- highway_layers = number of highway layers to use, if any

    -- there will be 2*n+1 inputs if using words or chars, 
    -- otherwise there will be 2*n + 2 inputs   
    local char_vec_layer, word_vec_layer, x, input_size_L, word_vec, char_vec
    local highway_layers = highway_layers or 0
    local length = length
    local inputs = {}
    if use_chars == 1 then
        table.insert(inputs, nn.Identity()()) -- batch_size x word length (char indices)
        char_vec_layer = LookupTable(char_vocab_size, char_vec_size)
        char_vec_layer.name = 'char_vecs' -- change name so we can refer to it easily later
    end
    if use_words == 1 then
        table.insert(inputs, nn.Identity()()) -- batch_size x 1 (word indices)
        word_vec_layer = LookupTable(word_vocab_size, word_vec_size)
        word_vec_layer.name = 'word_vecs' -- change name so we can refer to it easily later
    end

    local outputs = {}

    if use_chars == 1 then
        char_vec = char_vec_layer(inputs[1])
        local char_cnn = TDNN.tdnn(length, char_vec_size, feature_maps, kernels)
        char_cnn.name = 'cnn' -- change name so we can refer to it later
        local cnn_output = char_cnn(char_vec)
        input_size_L = torch.Tensor(feature_maps):sum()
        if use_words == 1 then
            word_vec = word_vec_layer(inputs[2])
            x = nn.JoinTable(2)({cnn_output, word_vec})
            input_size_L = input_size_L + word_vec_size
        else
            x = nn.Identity()(cnn_output)
        end
    else -- word_vecs only
        x = word_vec_layer(inputs[1])
        input_size_L = word_vec_size
    end

    if batch_norm == 1 then    
        x = nn.BatchNormalization(0)(x)
    end

    if highway_layers > 0 then
        local highway_mlp = HighwayMLP.mlp(input_size_L, highway_layers)
        highway_mlp.name = 'highway'
        x = highway_mlp(x)
    end
    local x_ = nn.Linear(input_size_L, input_encoding_size)(x)
    table.insert(outputs, x_)
    return nn.gModule(inputs, outputs)
end

return LSTMTD

