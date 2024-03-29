export type Json =
    | string
    | number
    | boolean
    | null
    | { [key: string]: Json | undefined }
    | Json[]

export interface Database {
    public: {
        Tables: {
            cards: {
                Row: {
                    category: string | null
                    culture: string | null
                    id: number
                    image_url: string | null
                    name: string
                }
                Insert: {
                    category?: string | null
                    culture?: string | null
                    id?: never
                    image_url?: string | null
                    name: string
                }
                Update: {
                    category?: string | null
                    culture?: string | null
                    id?: never
                    image_url?: string | null
                    name?: string
                }
                Relationships: []
            }
            challenges: {
                Row: {
                    challenged_id: number | null
                    challenger_id: number | null
                    id: number
                    lobby_id: number | null
                    result: string | null
                    timestamp: string | null
                }
                Insert: {
                    challenged_id?: number | null
                    challenger_id?: number | null
                    id?: never
                    lobby_id?: number | null
                    result?: string | null
                    timestamp?: string | null
                }
                Update: {
                    challenged_id?: number | null
                    challenger_id?: number | null
                    id?: never
                    lobby_id?: number | null
                    result?: string | null
                    timestamp?: string | null
                }
                Relationships: [
                    {
                        foreignKeyName: "challenges_challenged_id_fkey"
                        columns: ["challenged_id"]
                        isOneToOne: false
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "challenges_challenger_id_fkey"
                        columns: ["challenger_id"]
                        isOneToOne: false
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "challenges_lobby_id_fkey"
                        columns: ["lobby_id"]
                        isOneToOne: false
                        referencedRelation: "lobbies"
                        referencedColumns: ["id"]
                    }
                ]
            }
            chat_messages: {
                Row: {
                    content: string
                    id: number
                    lobby_id: number | null
                    sender_id: number | null
                    timestamp: string | null
                }
                Insert: {
                    content: string
                    id?: never
                    lobby_id?: number | null
                    sender_id?: number | null
                    timestamp?: string | null
                }
                Update: {
                    content?: string
                    id?: never
                    lobby_id?: number | null
                    sender_id?: number | null
                    timestamp?: string | null
                }
                Relationships: [
                    {
                        foreignKeyName: "chat_messages_lobby_id_fkey"
                        columns: ["lobby_id"]
                        isOneToOne: false
                        referencedRelation: "lobbies"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "chat_messages_sender_id_fkey"
                        columns: ["sender_id"]
                        isOneToOne: false
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    }
                ]
            }
            completed_sets: {
                Row: {
                    culture: string | null
                    id: number
                    lobby_id: number | null
                    player_id: number | null
                }
                Insert: {
                    culture?: string | null
                    id?: never
                    lobby_id?: number | null
                    player_id?: number | null
                }
                Update: {
                    culture?: string | null
                    id?: never
                    lobby_id?: number | null
                    player_id?: number | null
                }
                Relationships: [
                    {
                        foreignKeyName: "completed_sets_lobby_id_fkey"
                        columns: ["lobby_id"]
                        isOneToOne: false
                        referencedRelation: "lobbies"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "completed_sets_player_id_fkey"
                        columns: ["player_id"]
                        isOneToOne: false
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    }
                ]
            }
            game_actions: {
                Row: {
                    action_type: string | null
                    card_id: number | null
                    id: number
                    lobby_id: number | null
                    player_id: number | null
                    target_player_id: number | null
                    timestamp: string | null
                }
                Insert: {
                    action_type?: string | null
                    card_id?: number | null
                    id?: never
                    lobby_id?: number | null
                    player_id?: number | null
                    target_player_id?: number | null
                    timestamp?: string | null
                }
                Update: {
                    action_type?: string | null
                    card_id?: number | null
                    id?: never
                    lobby_id?: number | null
                    player_id?: number | null
                    target_player_id?: number | null
                    timestamp?: string | null
                }
                Relationships: [
                    {
                        foreignKeyName: "game_actions_card_id_fkey"
                        columns: ["card_id"]
                        isOneToOne: false
                        referencedRelation: "cards"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "game_actions_lobby_id_fkey"
                        columns: ["lobby_id"]
                        isOneToOne: false
                        referencedRelation: "lobbies"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "game_actions_player_id_fkey"
                        columns: ["player_id"]
                        isOneToOne: false
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "game_actions_target_player_id_fkey"
                        columns: ["target_player_id"]
                        isOneToOne: false
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    }
                ]
            }
            game_states: {
                Row: {
                    game_data: Json | null
                    id: number
                    lobby_id: number | null
                }
                Insert: {
                    game_data?: Json | null
                    id?: never
                    lobby_id?: number | null
                }
                Update: {
                    game_data?: Json | null
                    id?: never
                    lobby_id?: number | null
                }
                Relationships: [
                    {
                        foreignKeyName: "game_states_lobby_id_fkey"
                        columns: ["lobby_id"]
                        isOneToOne: false
                        referencedRelation: "lobbies"
                        referencedColumns: ["id"]
                    }
                ]
            }
            lobbies: {
                Row: {
                    created_at: string | null
                    game_mode: string | null
                    host_id: number | null
                    id: number
                    status: string | null
                }
                Insert: {
                    created_at?: string | null
                    game_mode?: string | null
                    host_id?: number | null
                    id?: never
                    status?: string | null
                }
                Update: {
                    created_at?: string | null
                    game_mode?: string | null
                    host_id?: number | null
                    id?: never
                    status?: string | null
                }
                Relationships: [
                    {
                        foreignKeyName: "lobbies_host_id_fkey"
                        columns: ["host_id"]
                        isOneToOne: false
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    }
                ]
            }
            player_cards: {
                Row: {
                    card_id: number | null
                    lobby_id: number | null
                    location: string | null
                    player_id: number | null
                }
                Insert: {
                    card_id?: number | null
                    lobby_id?: number | null
                    location?: string | null
                    player_id?: number | null
                }
                Update: {
                    card_id?: number | null
                    lobby_id?: number | null
                    location?: string | null
                    player_id?: number | null
                }
                Relationships: [
                    {
                        foreignKeyName: "player_cards_card_id_fkey"
                        columns: ["card_id"]
                        isOneToOne: false
                        referencedRelation: "cards"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "player_cards_lobby_id_fkey"
                        columns: ["lobby_id"]
                        isOneToOne: false
                        referencedRelation: "lobbies"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "player_cards_player_id_fkey"
                        columns: ["player_id"]
                        isOneToOne: false
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    }
                ]
            }
            players_in_lobbies: {
                Row: {
                    lobby_id: number
                    player_id: number
                }
                Insert: {
                    lobby_id: number
                    player_id: number
                }
                Update: {
                    lobby_id?: number
                    player_id?: number
                }
                Relationships: [
                    {
                        foreignKeyName: "players_in_lobbies_lobby_id_fkey"
                        columns: ["lobby_id"]
                        isOneToOne: false
                        referencedRelation: "lobbies"
                        referencedColumns: ["id"]
                    },
                    {
                        foreignKeyName: "players_in_lobbies_player_id_fkey"
                        columns: ["player_id"]
                        isOneToOne: false
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    }
                ]
            }
            profiles: {
                Row: {
                    avatar_url: string | null
                    full_name: string | null
                    id: string
                    updated_at: string | null
                    username: string | null
                    website: string | null
                }
                Insert: {
                    avatar_url?: string | null
                    full_name?: string | null
                    id: string
                    updated_at?: string | null
                    username?: string | null
                    website?: string | null
                }
                Update: {
                    avatar_url?: string | null
                    full_name?: string | null
                    id?: string
                    updated_at?: string | null
                    username?: string | null
                    website?: string | null
                }
                Relationships: [
                    {
                        foreignKeyName: "profiles_id_fkey"
                        columns: ["id"]
                        isOneToOne: true
                        referencedRelation: "users"
                        referencedColumns: ["id"]
                    }
                ]
            }
            trivia_questions: {
                Row: {
                    answer: string
                    id: number
                    options: string[] | null
                    question: string
                }
                Insert: {
                    answer: string
                    id?: never
                    options?: string[] | null
                    question: string
                }
                Update: {
                    answer?: string
                    id?: never
                    options?: string[] | null
                    question?: string
                }
                Relationships: []
            }
            users: {
                Row: {
                    email: string
                    id: number
                    password: string
                    username: string
                }
                Insert: {
                    email: string
                    id?: never
                    password: string
                    username: string
                }
                Update: {
                    email?: string
                    id?: never
                    password?: string
                    username?: string
                }
                Relationships: []
            }
        }
        Views: {
            [_ in never]: never
        }
        Functions: {
            [_ in never]: never
        }
        Enums: {
            [_ in never]: never
        }
        CompositeTypes: {
            [_ in never]: never
        }
    }
}
